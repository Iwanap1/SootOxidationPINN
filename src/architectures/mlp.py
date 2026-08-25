from typing import Dict, Optional
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, cfg: Dict):
        super().__init__()

        layers = []
        prev = input_dim

        activation_name = cfg.get("activation")
        activation_cls = getattr(nn, activation_name) if activation_name else None
        dropout = cfg.get("dropout", 0.0)

        for h in cfg.get("hidden_dim", []):
            layers.append(nn.Linear(prev, h))

            if activation_cls is not None:
                layers.append(activation_cls())

            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev = h

        layers.append(nn.Linear(prev, cfg["output_dim"]))

        self.net = nn.Sequential(*layers)
        self.output_dim = cfg["output_dim"]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)