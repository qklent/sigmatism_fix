"""Timestamp-based masking for hard-S inpainting.

Converts ``hard_s_timestamps`` (list of [start, end] float pairs in seconds)
into a binary mel-spectrogram frame mask suitable for the inpainting model.

Frame index formula::

    frame_idx = int(time_sec * sample_rate / hop_length)

The mask tensor has shape ``(1, 1, T_frames)`` where ``1.0`` marks frames to
be inpainted (masked) and ``0.0`` marks frames to keep.
"""

from __future__ import annotations

import torch


def generate_mask(
    hard_s_timestamps: list[list[float]],
    num_frames: int,
    sample_rate: int = 22050,
    hop_length: int = 256,
    padding_ms: float = 10.0,
) -> torch.Tensor:
    """Create a binary frame-level mask from hard-S timestamp intervals.

    Parameters
    ----------
    hard_s_timestamps:
        List of ``[start, end]`` pairs in seconds indicating regions that
        should be masked (inpainted).
    num_frames:
        Total number of mel-spectrogram frames in the sequence.
    sample_rate:
        Audio sample rate in Hz.
    hop_length:
        Hop length (in samples) used when computing the spectrogram.
    padding_ms:
        Amount of padding (in milliseconds) to add on each side of every
        timestamp interval.  Helps cover transition artefacts at segment
        boundaries.

    Returns
    -------
    torch.Tensor
        Boolean-style float mask of shape ``(1, 1, num_frames)`` where
        ``1.0`` = masked (to inpaint) and ``0.0`` = keep.
    """
    mask = torch.zeros(num_frames, dtype=torch.float32)

    padding_sec = padding_ms / 1000.0

    for segment in hard_s_timestamps:
        start_sec = segment[0] - padding_sec
        end_sec = segment[1] + padding_sec

        start_frame = int(start_sec * sample_rate / hop_length)
        end_frame = int(end_sec * sample_rate / hop_length)

        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)

        if start_frame < end_frame:
            mask[start_frame:end_frame] = 1.0

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_frames)
