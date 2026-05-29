"""Unit tests for the OmniVoice TTS adapter — covers return-shape normalization."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.config.schema import OmniVoiceConfig


def _install_fake_omnivoice(generate_returns):
    """Inject a fake `omnivoice` module so the import inside __init__ resolves."""
    fake_omnivoice = types.ModuleType("omnivoice")
    mock_model = MagicMock()
    mock_model.generate.return_value = generate_returns
    fake_omnivoice.OmniVoice = MagicMock()
    fake_omnivoice.OmniVoice.from_pretrained.return_value = mock_model
    sys.modules["omnivoice"] = fake_omnivoice
    return mock_model


def _build_adapter(generate_returns):
    mock_model = _install_fake_omnivoice(generate_returns)
    from src.inference.omnivoice_adapter import OmniVoiceAdapter

    adapter = OmniVoiceAdapter(OmniVoiceConfig(), device="cpu")
    return adapter, mock_model


def test_synthesize_returns_tensor_directly():
    waveform = torch.zeros(48000)
    adapter, _ = _build_adapter(generate_returns=waveform)
    out, sr = adapter.synthesize("hi", "/fake/ref.wav", ref_text="hi")
    assert out.shape == (1, 48000)
    assert sr == adapter.cfg.sample_rate


def test_synthesize_handles_list_output_with_numpy_array():
    """OmniVoice may emit list[ndarray] — exercise both branches in one shot."""
    arr = np.zeros(24000, dtype=np.float32)
    adapter, _ = _build_adapter(generate_returns=[arr])
    out, sr = adapter.synthesize("hi", "/fake/ref.wav")
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 24000)
    assert sr == adapter.cfg.sample_rate


def test_synthesize_does_not_pass_ref_text_when_none():
    waveform = torch.zeros(1, 48000)
    adapter, mock_model = _build_adapter(generate_returns=waveform)
    adapter.synthesize("hi", "/fake/ref.wav", ref_text=None)
    kwargs = mock_model.generate.call_args.kwargs
    assert "ref_text" not in kwargs
    assert kwargs["ref_audio"] == "/fake/ref.wav"


def test_synthesize_forwards_ref_text_when_present():
    waveform = torch.zeros(1, 48000)
    adapter, mock_model = _build_adapter(generate_returns=waveform)
    adapter.synthesize("hi", "/fake/ref.wav", ref_text="reference")
    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs["ref_text"] == "reference"


def test_init_uses_bfloat16_dtype_when_configured():
    cfg = OmniVoiceConfig(dtype="bfloat16")
    _install_fake_omnivoice(generate_returns=torch.zeros(1, 1000))
    from src.inference.omnivoice_adapter import OmniVoiceAdapter

    adapter = OmniVoiceAdapter(cfg, device="cpu")
    assert adapter.cfg.dtype == "bfloat16"
