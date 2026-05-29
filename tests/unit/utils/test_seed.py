"""Unit tests for src.utils.seed."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_makes_random_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    set_seed(123)
    b = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert a == b


def test_set_seed_toggles_cudnn_determinism():
    set_seed(7)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
