"""
Utility functions for the repository.
"""

import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]  # type: ignore


def fix_jams():
    # Monkey patch NumPy to add back the removed float_ type alias
    if not hasattr(np, "float_"):
        np.float_ = np.float64


if __name__ == "__main__":
    fix_jams()
