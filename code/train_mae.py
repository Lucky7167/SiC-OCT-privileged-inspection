import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights, r3d_18
from common import CFG, ROIDataset, fit_normalization, load_manifest, set_seed


class TubeMAE(nn.Module):
    def __init__(self, weights=R3D_18_Weights.KINETICS400_V1):
        super().__init__()
        backbone = r3d_18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=4), nn.GELU(),
            nn.ConvTranspose3d(256, 64, kernel_size=4, stride=2), nn.GELU(),
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2),
        )

    @staticmethod
    def tube_mask(volume):
        batch, _, _, height, width = volume.shape
        grid = 8
        ratios = torch.empty(batch, 1, 1).uniform_(0.75, 0.90)
        coarse = (torch.rand(batch, grid, grid) < ratios).to(volume.device)
        mask = coarse.repeat_interleave(height // grid, 1).repeat_interleave(width // grid, 2)
        mask = mask[:, None, None].expand_as(volume)
        return volume.masked_fill(mask, 0.5), mask

    def forward(self, volume):
        masked, mask = self.tube_mask(volume)
        latent = self.encoder(masked)
        reconstruction = self.decoder(latent)
        reconstruction = F.interpolate(
            reconstruction, size=volume.shape[-3:], mode="trilinear", align_corners=False,
        )
        squared_error = (reconstruction - volume[:, :1]) ** 2
        loss_mask = mask[:, :1].to(squared_error.dtype)
        return (squared_error * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def train(train_frame, output: Path, epochs=40, batch_size=8, num_workers=0):
    set_seed()
    print(
        "MAE numeric policy: "
        f"device={CFG.device}, "
        f"cudnn_tf32={torch.backends.cudnn.allow_tf32}, "
        f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32}",
        flush=True,
    )
    loader = DataLoader(
        ROIDataset(train_frame, fit_normalization(train_frame), load_volume=True),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
    )
    model = TubeMAE().to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    for epoch in range(epochs):
        epoch_losses = []
        model.train()
        for module in model.encoder.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()
        for batch_index, batch in enumerate(loader):
            volume = batch["volume"].to(CFG.device)
            for attempt in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = model(volume)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise FloatingPointError("non-finite MAE parameters after optimizer step")
                epoch_losses.append(loss.item())
                break
            else:
                raise FloatingPointError(
                    f"non-finite MAE loss or gradients after three mask retries "
                    f"at batch {batch_index}: {list(batch['roi_id'])}"
                )
        print(f"{output.parent.name}/mae epoch={epoch + 1}/{epochs} loss={sum(epoch_losses)/max(len(epoch_losses),1):.6f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-wafers", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    manifest = load_manifest()
    train(
        manifest[manifest.wafer.astype(str).isin(args.train_wafers)], args.output,
        args.epochs, args.batch_size, args.num_workers,
    )
