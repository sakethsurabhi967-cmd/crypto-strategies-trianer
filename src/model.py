"""GRU sequence classifier + save/load helpers for trained bundles.

A "bundle" (.pt file) is self-contained: weights, normalisation stats,
feature list and hyper-parameters — the predictor only needs the file.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class EmaTrendNet(nn.Module):
    """GRU over a window of EMA/momentum features -> P(down/flat/up)."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
        out, _ = self.gru(x)
        return self.head(out[:, -1])  # logits from the last timestep


def save_bundle(path: str, model: EmaTrendNet, meta: dict[str, Any]) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_bundle(path: str, device: str = "cpu") -> tuple[EmaTrendNet, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = EmaTrendNet(
        n_features=len(meta["features"]),
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta
