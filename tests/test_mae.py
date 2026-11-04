import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models.video import R3D_18_Weights
except ImportError:
    from enum import Enum
    import torchvision.models.video
    R3D_18_Weights = Enum("R3D_18_Weights", {"KINETICS400_V1": "KINETICS400_V1"})
    torchvision.models.video.R3D_18_Weights = R3D_18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import models
import train_mae


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Identity()
        self.layer1 = nn.Identity()
        self.avgpool = nn.Identity()
        self.fc = nn.Identity()


class FixedDecoder(nn.Module):
    def __init__(self, reconstruction):
        super().__init__()
        self.reconstruction = reconstruction

    def forward(self, latent):
        return self.reconstruction


def make_mae():
    with patch.object(train_mae, "r3d_18", return_value=FakeBackbone()):
        return train_mae.TubeMAE()


def test_r3d_18_uses_kinetics400_v1_by_default():
    with patch.object(models, "r3d_18", return_value=FakeBackbone()) as constructor:
        models.OCTEncoder()
        constructor.assert_called_once_with(weights=R3D_18_Weights.KINETICS400_V1)
    with patch.object(train_mae, "r3d_18", return_value=FakeBackbone()) as constructor:
        train_mae.TubeMAE()
        constructor.assert_called_once_with(weights=R3D_18_Weights.KINETICS400_V1)


def test_r3d_18_accepts_none_weights_for_tests():
    with patch.object(models, "r3d_18", return_value=FakeBackbone()) as constructor:
        models.OCTEncoder(weights=None)
        constructor.assert_called_once_with(weights=None)
    with patch.object(train_mae, "r3d_18", return_value=FakeBackbone()) as constructor:
        train_mae.TubeMAE(weights=None)
        constructor.assert_called_once_with(weights=None)


def test_mae_decoder_matches_three_transpose_convolutions():
    model = make_mae()
    layers = list(model.decoder)
    assert [type(layer) for layer in layers] == [
        nn.ConvTranspose3d, nn.GELU, nn.ConvTranspose3d, nn.GELU, nn.ConvTranspose3d,
    ]
    convolutions = layers[::2]
    assert [(layer.in_channels, layer.out_channels) for layer in convolutions] == [
        (512, 256), (256, 64), (64, 1),
    ]
    assert all(layer.kernel_size == (4, 4, 4) for layer in convolutions)
    assert [layer.stride for layer in convolutions] == [(4, 4, 4), (2, 2, 2), (2, 2, 2)]
    assert not any(isinstance(layer, nn.Sigmoid) for layer in model.decoder)


def test_mae_loss_aligns_reconstruction_and_uses_only_masked_voxels():
    model = make_mae()
    model.encoder = nn.Identity()
    reconstruction = torch.full((1, 1, 2, 4, 4), 0.25)
    model.decoder = FixedDecoder(reconstruction)
    volume = torch.arange(256, dtype=torch.float32).reshape(1, 1, 4, 8, 8) / 256
    mask = torch.zeros_like(volume, dtype=torch.bool)
    mask[..., 1:3, 2:6] = True
    masked = volume.masked_fill(mask, 0.5)
    with patch.object(train_mae.TubeMAE, "tube_mask", return_value=(masked, mask)):
        loss = model(volume)
    aligned = F.interpolate(
        reconstruction, size=volume.shape[-3:], mode="trilinear", align_corners=False,
    )
    loss_mask = mask[:, :1].float()
    expected = (((aligned - volume[:, :1]) ** 2) * loss_mask).sum() / loss_mask.sum()
    assert torch.allclose(loss, expected)


def test_tube_mask_is_constant_over_depth_and_channels():
    volume = torch.ones(2, 3, 4, 64, 64)
    masked, mask = train_mae.TubeMAE.tube_mask(volume)
    assert masked.shape == volume.shape
    assert mask.shape == volume.shape
    assert torch.equal(mask, mask[:, :, :1].expand_as(mask))
    assert torch.equal(mask, mask[:, :1].expand_as(mask))
    assert torch.equal(masked[mask], torch.full_like(masked[mask], 0.5))
    assert torch.equal(masked[~mask], volume[~mask])
