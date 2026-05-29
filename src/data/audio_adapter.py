"""
Audio I/O adapter for sigmatism_fix.

This module is the **single point of contact** for all audio loading and
saving operations. All other modules (fft.py, vocoder.py, dataset.py) MUST
go through this adapter instead of importing torchaudio or librosa directly.

By isolating the audio backend here, swapping implementations (e.g. from
torchaudio to librosa for a specific op) requires changing only this file
(constitution principle IV and Engineering Standards).

Supported backends (configure via adapter internals, not the caller):
  - torchaudio  — primary; GPU-aware, differentiable STFT
  - soundfile   — fallback for reading formats unsupported by torchaudio
  - librosa     — optional; useful for analysis notebooks, NOT for training
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio


def load_waveform(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load an audio file and return a mono waveform tensor.

    Parameters
    ----------
    path:
        Path to the audio file (WAV, FLAC, MP3, etc.).
    sample_rate:
        Target sample rate in Hz. The waveform is resampled if necessary.

    Returns
    -------
    torch.Tensor
        Shape ``(T,)`` float32 waveform, values in ``[-1.0, 1.0]``.
    """
    waveform, sr = torchaudio.load(str(path))
    # waveform shape: (channels, T)

    # Resample if the source sample rate differs from the target.
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Stereo (or multi-channel) → mono: average across channels.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Return shape (T,) float32.
    return waveform.squeeze(0).float()


def save_waveform(waveform: torch.Tensor, path: str | Path, sample_rate: int) -> None:
    """Save a waveform tensor to disk as a WAV file.

    Parameters
    ----------
    waveform:
        Shape ``(T,)`` float32 tensor.
    path:
        Destination file path (parent directory must exist).
    sample_rate:
        Sample rate of the waveform in Hz.
    """
    # torchaudio.save expects shape (channels, T).
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(str(path), waveform, sample_rate)


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """Compute a mel-spectrogram matching the original HiFi-GAN preprocessing.

    Replicates the exact mel computation from jik876/hifi-gan so that the
    pretrained vocoder receives spectrograms in the domain it was trained on.

    Key differences from torchaudio.transforms.MelSpectrogram:
      - center=False STFT with manual reflect padding
      - librosa mel filterbank (different coefficients)
      - magnitude via sqrt(real² + imag² + 1e-9)

    Parameters
    ----------
    waveform:
        Shape ``(T,)`` float32 mono waveform.
    sample_rate:
        Sample rate of the waveform in Hz.
    n_fft:
        FFT window size (also used as win_length).
    hop_length:
        Hop length in samples between STFT frames.
    n_mels:
        Number of mel filter-bank channels.
    f_max:
        Maximum frequency for the mel filterbank (Hz).

    Returns
    -------
    torch.Tensor
        Shape ``(1, n_mels, T_frames)`` linear mel-spectrogram.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    device = waveform.device

    # Build mel basis from librosa.
    # The speechbrain/tts-hifigan-ljspeech checkpoint was trained with
    # norm='slaney' (area-normalized) filterbanks. Using norm=None produces
    # log-mel values up to ~4.7 which the vocoder cannot invert (chopped audio).
    mel_fb = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=f_max,
        norm="slaney",
    )
    mel_basis = torch.from_numpy(mel_fb).float().to(device)

    # Hann window
    window = torch.hann_window(n_fft, device=device)

    # Reflect-pad then STFT with center=False (matches original HiFi-GAN)
    pad_size = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(
        waveform, (pad_size, pad_size), mode="reflect",
    )
    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    # Magnitude with epsilon inside sqrt (matches original)
    magnitudes = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)

    # Apply mel filterbank: (n_mels, n_fft//2+1) @ (B, n_fft//2+1, T) -> (B, n_mels, T)
    mel_spec = torch.matmul(mel_basis, magnitudes)

    return mel_spec


def linear_mel_to_log_normalized(
    mel: torch.Tensor,
    floor: float = 1e-5,
    norm_min: float = -11.5,
    norm_max: float = 2.0,
) -> torch.Tensor:
    """Convert linear mel-spectrogram to log domain and normalize to [0, 1].

    This compresses the dynamic range so that high-frequency sibilant bands
    (which have small linear values) contribute equally to the loss compared
    to loud low-frequency bands.

    Parameters
    ----------
    mel:
        Linear mel-spectrogram, any shape.
    floor:
        Small constant added before log to avoid -inf.
    norm_min:
        Value mapped to 0.0 (approximately log(floor)).
    norm_max:
        Value mapped to 1.0 (approximately log of loud speech mel values).

    Returns
    -------
    torch.Tensor
        Normalized log-mel in [0, 1] range (clamped), same shape as input.
    """
    log_mel = torch.log(torch.clamp(mel, min=floor))
    normalized = (log_mel - norm_min) / (norm_max - norm_min)
    return normalized.clamp(0.0, 1.0)


def normalized_mel_to_log_mel(
    normalized: torch.Tensor,
    norm_min: float = -11.5,
    norm_max: float = 2.0,
) -> torch.Tensor:
    """Convert [0,1]-normalized mel directly to raw log-mel for HiFi-GAN input.

    Shortcut that skips the linear mel intermediate:
    ``raw_log_mel = normalized * (norm_max - norm_min) + norm_min``

    Parameters
    ----------
    normalized:
        Normalized log-mel in [0, 1] range, any shape.
    norm_min:
        Value mapped to 0.0 (approximately log(floor)).
    norm_max:
        Value mapped to 1.0 (approximately log of loud speech mel values).

    Returns
    -------
    torch.Tensor
        Raw log-mel spectrogram, same shape as input.
    """
    return normalized * (norm_max - norm_min) + norm_min


def log_normalized_to_linear_mel(
    normalized: torch.Tensor,
    floor: float = 1e-5,
    norm_min: float = -11.5,
    norm_max: float = 2.0,
) -> torch.Tensor:
    """Invert log-normalized mel back to linear domain for vocoding.

    Parameters
    ----------
    normalized:
        Normalized log-mel in [0, 1] range, any shape.
    floor:
        Same floor used in the forward transform.
    norm_min:
        Same norm_min used in the forward transform.
    norm_max:
        Same norm_max used in the forward transform.

    Returns
    -------
    torch.Tensor
        Linear mel-spectrogram, same shape as input.
    """
    log_mel = normalized * (norm_max - norm_min) + norm_min
    return torch.exp(log_mel) - floor
