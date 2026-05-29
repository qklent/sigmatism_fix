"""Unit tests for src.utils.config (loader + validator)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config import load_config, validate_config


@pytest.fixture
def minimal_cfg_dict():
    return {
        "seed": 1,
        "precision": "fp32",
        "device": "cpu",
        "sanity_check": False,
        "data": {
            "dataset_name": "x",
            "sample_rate": 22050,
            "n_mels": 80,
            "n_fft": 1024,
            "hop_length": 256,
            "mask_padding_ms": 10.0,
            "split_train": "train",
            "split_val": "validation",
        },
        "model": {"type": "UNet", "in_channels": 2, "out_channels": 1},
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "lr": 1e-3,
            "precision": "fp32",
            "checkpoint_every_n_epochs": 1,
        },
        "loss": {
            "type": "SpectralL1",
            "masked_l1_weight": 1.0,
            "multiscale_spectral_weight": 0.5,
            "unmasked_l1_weight": 0.01,
        },
        "wandb": {"project": "t", "entity": ""},
    }


def test_validate_config_returns_pydantic_model(minimal_cfg_dict):
    cfg = validate_config(minimal_cfg_dict)
    assert cfg.seed == 1
    assert cfg.precision == "fp32"


def test_validate_config_raises_on_missing_precision(minimal_cfg_dict):
    del minimal_cfg_dict["precision"]
    with pytest.raises(ValidationError):
        validate_config(minimal_cfg_dict)


def test_load_config_reads_yaml(tmp_path: Path, minimal_cfg_dict):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(minimal_cfg_dict))
    cfg = load_config(p)
    assert cfg.seed == 1


def test_load_config_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
