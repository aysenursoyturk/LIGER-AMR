"""Frozen neural architectures used in the LIGER-AMR final 40-fold evaluation.

Source: liger_amr_final_40fold(6).py (user-provided Colab export).
This module contains model definitions only; it does not train models or load data.
Layer/module names are retained for state_dict compatibility.
"""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Shared 256 -> 128 encoder used in the final comparison."""

    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleView(nn.Module):
    """Final SNP-only or Gene-only baseline (select the input at call time)."""

    def __init__(self, d: int):
        super().__init__()
        self.enc = Encoder(d, 128)
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x)).squeeze(1)


class Concat(nn.Module):
    """Final concatenation baseline, without adaptive gating."""

    def __init__(self, ds: int, dg: int):
        super().__init__()
        self.se = Encoder(ds, 128)
        self.ge = Encoder(dg, 128)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, xs: torch.Tensor, xg: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.se(xs), self.ge(xg)], dim=1)).squeeze(1)


class LIGER_AMR_v2(nn.Module):
    """Final feature-wise gated residual fusion architecture.

    Returns a dictionary with logits, SNP/gene auxiliary logits, and the
    sample-specific, per-latent-dimension sigmoid gate (shape [batch, 128]).
    """

    def __init__(self, snp_dim: int, gene_dim: int, latent_dim: int = 128):
        super().__init__()
        self.snp_encoder = Encoder(snp_dim, latent_dim)
        self.gene_encoder = Encoder(gene_dim, latent_dim)
        self.snp_aux = nn.Linear(latent_dim, 1)
        self.gene_aux = nn.Linear(latent_dim, 1)
        self.feature_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, xs: torch.Tensor, xg: torch.Tensor) -> dict[str, torch.Tensor]:
        zs = self.snp_encoder(xs)
        zg = self.gene_encoder(xg)
        gate = self.feature_gate(torch.cat([zs, zg], dim=1))
        z = gate * zs + (1.0 - gate) * zg
        return {
            "logits": self.fusion_head(torch.cat([z, zs, zg], dim=1)).squeeze(1),
            "snp_logits": self.snp_aux(zs).squeeze(1),
            "gene_logits": self.gene_aux(zg).squeeze(1),
            "gate": gate,
        }


def liger_loss(
    output: dict[str, torch.Tensor],
    targets: torch.Tensor,
    lambda_snp: float = 0.20,
    lambda_gene: float = 0.20,
) -> torch.Tensor:
    """Final training objective: main BCE + 0.20 SNP BCE + 0.20 gene BCE."""
    criterion = nn.BCEWithLogitsLoss()
    return (
        criterion(output["logits"], targets)
        + lambda_snp * criterion(output["snp_logits"], targets)
        + lambda_gene * criterion(output["gene_logits"], targets)
    )


def build_model(name: str, snp_dim: int, gene_dim: int) -> nn.Module:
    """Instantiate one of the four frozen architectures by experiment name."""
    if name == "SNP-only":
        return SingleView(snp_dim)
    if name == "Gene-only":
        return SingleView(gene_dim)
    if name == "Concat":
        return Concat(snp_dim, gene_dim)
    if name == "LIGER-AMR-v2":
        return LIGER_AMR_v2(snp_dim, gene_dim, 128)
    raise ValueError(f"Unknown model: {name!r}")


__all__ = ["Encoder", "SingleView", "Concat", "LIGER_AMR_v2", "liger_loss", "build_model"]
