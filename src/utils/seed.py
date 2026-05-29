"""
Random seed utilities for sigmatism_fix.

All experiments MUST call ``set_seed`` at startup before any data loading,
model initialisation, or augmentation. This ensures bit-for-bit reproducibility
across runs with the same config (constitution principle I).

Seeded targets
--------------
- Python built-in ``random`` module
- NumPy global random state
- PyTorch CPU and CUDA random states
- ``torch.backends.cudnn.deterministic = True`` (may reduce throughput slightly)

The seed value is read from ``cfg.seed`` and defaults to ``42`` if not set.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all random number generators for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value. Use ``cfg.seed`` (default 42) from the experiment
        config to ensure reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
