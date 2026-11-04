from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from common import CFG


class OCTEncoder(nn.Module):
    def __init__(self, mae_checkpoint=None, weights=R3D_18_Weights.KINETICS400_V1):
        super().__init__()
        backbone = r3d_18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        if mae_checkpoint is not None:
            state = torch.load(mae_checkpoint, map_location="cpu", weights_only=True)
            self.features.load_state_dict(state)

    def forward(self, volume):
        return F.adaptive_avg_pool3d(self.features(volume), 1).flatten(1)


class Teacher(nn.Module):
    def __init__(self, mae_checkpoint=None):
        super().__init__()
        self.oct_encoder = OCTEncoder(mae_checkpoint)
        self.local_projection = nn.Linear(CFG.dino_dim, CFG.feature_dim)
        self.context_projection = nn.Linear(CFG.dino_dim, CFG.feature_dim)
        self.volume_projection = nn.Linear(512, CFG.feature_dim)
        self.metadata_projection = nn.Sequential(
            nn.Linear(CFG.metadata_dim, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, CFG.feature_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=CFG.feature_dim,
            nhead=4,
            dim_feedforward=1024,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=1)
        self.query = nn.Parameter(torch.randn(1, 1, CFG.feature_dim))
        self.pool = nn.MultiheadAttention(CFG.feature_dim, 4, batch_first=True)
        self.class_head = nn.Linear(CFG.feature_dim, 3)
        self.depth_evidence_head = nn.Linear(CFG.feature_dim, 1)
        self.score_head = nn.Linear(CFG.feature_dim, 1)

    def forward(self, local, context, volume, metadata):
        tokens = torch.stack([
            self.local_projection(local),
            self.context_projection(context),
            self.volume_projection(self.oct_encoder(volume)),
            self.metadata_projection(metadata),
        ], dim=1)
        tokens = self.fusion(tokens)
        query = self.query.expand(tokens.shape[0], -1, -1)
        feature = self.pool(query, tokens, tokens, need_weights=False)[0].squeeze(1)
        return {
            "feature": feature,
            "logits": self.class_head(feature),
            "q": torch.sigmoid(self.depth_evidence_head(feature)).squeeze(-1),
            "score": torch.sigmoid(self.score_head(feature)).squeeze(-1),
        }


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 2 * CFG.dino_dim + CFG.metadata_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, CFG.feature_dim), nn.GELU(),
            nn.Linear(CFG.feature_dim, CFG.feature_dim), nn.GELU(),
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(CFG.feature_dim, CFG.feature_dim),
            nn.LayerNorm(CFG.feature_dim),
        )
        self.class_head = nn.Linear(CFG.feature_dim, 3)
        self.score_head = nn.Linear(CFG.feature_dim, 1)

    def forward(self, local, context, metadata):
        feature = self.encoder(torch.cat([local, context, metadata], dim=-1))
        return {
            "feature": feature,
            "projected_feature": self.feature_projection(feature),
            "logits": self.class_head(feature),
            "score": torch.sigmoid(self.score_head(feature)).squeeze(-1),
        }


class MicroscopyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 2 * CFG.dino_dim + CFG.metadata_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, CFG.feature_dim), nn.GELU(),
            nn.Linear(CFG.feature_dim, CFG.feature_dim), nn.GELU(),
        )
        self.class_head = nn.Linear(CFG.feature_dim, 3)

    def forward(self, local, context, metadata):
        feature = self.encoder(torch.cat([local, context, metadata], dim=-1))
        return {"feature": feature, "logits": self.class_head(feature)}
