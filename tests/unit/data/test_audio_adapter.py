"""Unit tests for the audio I/O adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchaudio

from src.data.audio_adapter import (
    compute_mel_spectrogram,
    linear_mel_to_log_normalized,
    load_waveform,
    log_normalized_to_linear_mel,
    normalized_mel_to_log_mel,
    save_waveform,
)


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    sr = 16000
    t = torch.linspace(0, 1.0, sr)
    wave = (0.1 * torch.sin(2 * torch.pi * 440 * t)).unsqueeze(0)
    p = tmp_path / "tone.wav"
    torchaudio.save(str(p), wave, sr)
    return p


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    sr = 16000
    left = 0.1 * torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1.0, sr))
    right = 0.1 * torch.sin(2 * torch.pi * 880 * torch.linspace(0, 1.0, sr))
    wave = torch.stack([left, right], dim=0)
    p = tmp_path / "stereo.wav"
    torchaudio.save(str(p), wave, sr)
    return p


def test_load_waveform_returns_1d_float32(tone_wav: Path):
    out = load_waveform(tone_wav, sample_rate=16000)
    assert out.dim() == 1
    assert out.dtype == torch.float32
    assert out.shape[0] == 16000


def test_load_waveform_resamples(tone_wav: Path):
    out = load_waveform(tone_wav, sample_rate=8000)
    assert out.shape[0] == 8000


def test_load_waveform_downmixes_stereo_to_mono(stereo_wav: Path):
    out = load_waveform(stereo_wav, sample_rate=16000)
    assert out.dim() == 1
    assert out.shape[0] == 16000


def test_save_waveform_accepts_1d_tensor(tmp_path: Path):
    wave = torch.zeros(16000)
    out = tmp_path / "out.wav"
    save_waveform(wave, out, sample_rate=16000)
    assert out.exists()
    re_wave, sr = torchaudio.load(str(out))
    assert sr == 16000
    assert re_wave.shape == (1, 16000)


def test_save_waveform_accepts_2d_tensor(tmp_path: Path):
    wave = torch.zeros(1, 16000)
    out = tmp_path / "out.wav"
    save_waveform(wave, out, sample_rate=16000)
    assert out.exists()


def test_mel_spectrogram_shape():
    wave = torch.zeros(22050)
    mel = compute_mel_spectrogram(wave, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80)
    assert mel.shape[0] == 1
    assert mel.shape[1] == 80
    assert mel.shape[2] > 0


def test_mel_spectrogram_accepts_2d_input():
    wave = torch.zeros(1, 22050)
    mel = compute_mel_spectrogram(wave)
    assert mel.dim() == 3


def test_linear_mel_log_normalized_round_trip():
    mel = torch.rand(1, 80, 50) * 5.0
    norm = linear_mel_to_log_normalized(mel)
    assert (norm >= 0.0).all() and (norm <= 1.0).all()
    back = log_normalized_to_linear_mel(norm)
    # Within the clamping range we should recover something positive.
    assert back.shape == mel.shape


def test_normalized_mel_to_log_mel():
    norm = torch.tensor([0.0, 0.5, 1.0])
    log = normalized_mel_to_log_mel(norm, norm_min=-11.5, norm_max=2.0)
    assert torch.allclose(log[0], torch.tensor(-11.5))
    assert torch.allclose(log[2], torch.tensor(2.0))
