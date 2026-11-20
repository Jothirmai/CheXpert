"""Stage 1: Disease-aware Visual-Textual Alignment + Prior-Guided Attention
Module (PrAM) for multi-view dependency capture.

Architecture:
  - CNN encoder (DenseNet-121 backbone) -> visual feature map
  - Disease embedding table -> per-disease query vectors
  - Cross-attention: disease queries attend to visual features
  - PrAM: prior-guided attention across current + prior views
  - Output: aligned disease-aware visual representations + disease scores
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv


class PrAM(nn.Module):
    """Prior-Guided Attention Module.

    Computes attention over visual features guided by a prior (disease
    embedding). Supports multi-view by attending across a stack of views
    (current + prior scans).
    """

    def __init__(self, visual_dim: int, disease_dim: int, num_heads: int = 4):
        super().__init__()
        self.visual_dim = visual_dim
        self.disease_dim = disease_dim
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads

        self.q_proj = nn.Linear(disease_dim, visual_dim)
        self.k_proj = nn.Linear(visual_dim, visual_dim)
        self.v_proj = nn.Linear(visual_dim, visual_dim)
        self.out_proj = nn.Linear(visual_dim, visual_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        visual_feats: torch.Tensor,   # [B, N, visual_dim] or [B, C, H, W]
        disease_query: torch.Tensor,  # [B, disease_dim]
        prior_mask: Optional[torch.Tensor] = None,  # [B, N] bool, True=valid view
    ) -> torch.Tensor:
        # Flatten spatial if 4D
        if visual_feats.dim() == 4:
            B, C, H, W = visual_feats.shape
            visual_feats = visual_feats.flatten(2).transpose(1, 2)  # [B, N, C]

        B, N, C = visual_feats.shape

        single_query = disease_query.dim() == 2
        queries = disease_query.unsqueeze(1) if single_query else disease_query
        query_count = queries.shape[1]
        q = self.q_proj(queries)
        k = self.k_proj(visual_feats)                # [B, N, C]
        v = self.v_proj(visual_feats)                # [B, N, C]

        # Multi-head reshape
        q = q.view(B, query_count, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, 1, d]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, h, 1, N]

        if prior_mask is not None:
            # prior_mask: [B, N] -> [B, 1, 1, N]
            mask = prior_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, h, 1, d]
        out = out.transpose(1, 2).contiguous().view(B, query_count, C)  # [B, 1, C]
        out = self.out_proj(out)
        return out.squeeze(1) if single_query else out


class DiseaseAwareAlignment(nn.Module):
    """Disease-aware Visual-Textual Alignment.

    Aligns visual features with disease text embeddings using cross-attention.
    Produces per-disease confidence scores and aligned representations.
    """

    def __init__(
        self,
        num_diseases: int,
        visual_dim: int = 1024,
        disease_dim: int = 256,
        pram_heads: int = 4,
    ):
        super().__init__()
        self.num_diseases = num_diseases
        # Legacy checkpoints use pooled queries; new candidates opt into per-finding queries.
        self.disease_specific_queries = False
        self.visual_dim = visual_dim
        self.disease_dim = disease_dim

        # Disease embedding table (learnable disease queries)
        self.disease_embeddings = nn.Embedding(num_diseases, disease_dim)

        # Disease classifier head
        self.disease_classifier = nn.Linear(visual_dim, num_diseases)

        # PrAM for multi-view attention
        self.pram = PrAM(visual_dim, disease_dim, num_heads=pram_heads)

        # Disease-to-visual projection for alignment loss
        self.disease_to_visual = nn.Linear(disease_dim, visual_dim)

    def forward(
        self,
        visual_feats: torch.Tensor,   # [B, C, H, W] or [B, N, C]
        disease_ids: Optional[torch.Tensor] = None,  # [B] or None
        prior_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        B = visual_feats.shape[0]

        # Default disease_ids: all diseases for each sample
        if disease_ids is None:
            disease_ids = torch.arange(self.num_diseases, device=visual_feats.device)
            disease_ids = disease_ids.unsqueeze(0).expand(B, -1)  # [B, D]
        else:
            disease_ids = disease_ids.unsqueeze(1).expand(-1, self.num_diseases)  # [B, D]

        # Flatten visual to [B, N, C]
        if visual_feats.dim() == 4:
            B, C, H, W = visual_feats.shape
            visual_flat = visual_feats.flatten(2).transpose(1, 2)  # [B, N, C]
        else:
            visual_flat = visual_feats  # [B, N, C]

        # Disease queries: [B, D, disease_dim]
        disease_queries = self.disease_embeddings(disease_ids)

        # Align each disease query with visual features via PrAM
        aligned = self.pram(visual_flat, disease_queries, prior_mask=prior_mask)

        # Disease scores
        if self.disease_specific_queries:
            disease_logits = (aligned * self.disease_classifier.weight.unsqueeze(0)).sum(-1)
            if self.disease_classifier.bias is not None:
                disease_logits = disease_logits + self.disease_classifier.bias
        else:
            disease_logits = self.disease_classifier(aligned.mean(dim=1))  # [B, D]

        # Alignment similarity: disease embedding vs aligned visual
        disease_proj = self.disease_to_visual(disease_queries)  # [B, D, visual_dim]
        alignment_sim = F.cosine_similarity(
            disease_proj, aligned, dim=-1
        )  # [B, D]

        return {
            "disease_logits": disease_logits,
            "disease_probs": torch.sigmoid(disease_logits),
            "aligned_feats": aligned,          # [B, D, visual_dim]
            "alignment_sim": alignment_sim,    # [B, D]
            "disease_queries": disease_queries, # [B, D, disease_dim]
        }


class Stage1CNN(nn.Module):
    """Stage 1 full model: CNN encoder + disease-aware alignment.

    Backbone: torchxrayvision DenseNet-121 (feature extractor).
    Output: disease logits + aligned features for Stage 2.
    """

    def __init__(self, num_diseases: int = 14, pretrained: bool = True, pretrained_weights: str = "densenet121-res224-all"):
        super().__init__()
        self.num_diseases = num_diseases

        # CNN backbone (feature extractor, no classifier head)
        if pretrained:
            backbone = xrv.models.DenseNet(weights=pretrained_weights)
        else:
            backbone = xrv.models.DenseNet(weights=None)

        # Remove the classifier to get features
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        # DenseNet feature output: [B, 1024, 7, 7] for 224 input

        self.visual_dim = 1024

        # Disease-aware alignment module
        self.alignment = DiseaseAwareAlignment(
            num_diseases=num_diseases,
            visual_dim=self.visual_dim,
            disease_dim=256,
            pram_heads=4,
        )

    def forward(
        self,
        images: torch.Tensor,          # [B, 1, H, W] grayscale X-ray
        disease_ids: Optional[torch.Tensor] = None,
        prior_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # CNN features: [B, 1024, 7, 7]
        feats = self.encoder(images)

        # Disease-aware alignment
        out = self.alignment(feats, disease_ids=disease_ids, prior_mask=prior_mask)
        out["visual_feats"] = feats
        return out

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, num_diseases: int = 14, pretrained: bool = False, device: str = "cpu"):
        model = cls(num_diseases=num_diseases, pretrained=pretrained)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
