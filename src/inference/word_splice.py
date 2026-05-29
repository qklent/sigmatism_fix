"""Word-level audio splicing for resynthesis pipeline.

Replaces target words in original audio with corresponding words from
resynthesized audio, applying Hann crossfade at boundaries.
"""

from __future__ import annotations

import logging

import torch
import torchaudio.functional as F

from src.preprocessing.forced_aligner import WordAlignment

logger = logging.getLogger(__name__)


def _hann_crossfade(length: int, device: torch.device) -> torch.Tensor:
    """Create a Hann fade-in window of given length."""
    if length <= 0:
        return torch.ones(0, device=device)
    return 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, length, device=device)))


def splice_words(
    original_waveform: torch.Tensor,
    original_sr: int,
    original_words: list[WordAlignment],
    resynth_waveform: torch.Tensor,
    resynth_sr: int,
    resynth_words: list[WordAlignment],
    target_word_indices: list[int],
    crossfade_ms: float = 10.0,
    max_overlap_ms: float = 50.0,
    guard_ms: float = 0.0,
) -> torch.Tensor:
    """Replace target words in original audio with resynthesized versions.

    Args:
        original_waveform: Original audio tensor, shape (1, T) or (T,).
        original_sr: Sample rate of original audio.
        original_words: Word alignments from MFA on original audio.
        resynth_waveform: Resynthesized audio tensor, shape (1, T) or (T,).
        resynth_sr: Sample rate of resynthesized audio.
        resynth_words: Word alignments from MFA on resynthesized audio.
        target_word_indices: Indices of words to replace.
        crossfade_ms: Crossfade duration at splice boundaries (ms).
        max_overlap_ms: Max allowed overlap before truncating resynth word (ms).
        guard_ms: Symmetric pad added to each word region (ms). Absorbs aligner
            timing noise — set ~80 for GigaAM RNN-T (40 ms frame stride + late
            emission bias), keep at 0 for MFA.

    Returns:
        Modified waveform with target words replaced, shape (1, T).
    """
    # Ensure 2D
    if original_waveform.dim() == 1:
        original_waveform = original_waveform.unsqueeze(0)
    if resynth_waveform.dim() == 1:
        resynth_waveform = resynth_waveform.unsqueeze(0)

    # Resample resynthesized audio to original SR if needed
    if resynth_sr != original_sr:
        resynth_waveform = F.resample(resynth_waveform, resynth_sr, original_sr)

    device = original_waveform.device
    crossfade_samples = int(crossfade_ms * original_sr / 1000)
    guard_samples = int(guard_ms * original_sr / 1000)

    # Build output by copying original and splicing in replacements
    # Process words from right to left so sample indices stay valid
    output = original_waveform.clone()

    for idx in sorted(target_word_indices, reverse=True):
        if idx >= len(original_words) or idx >= len(resynth_words):
            logger.warning("Word index %d out of range, skipping", idx)
            continue

        orig_word = original_words[idx]
        resynth_word = resynth_words[idx]

        orig_start = max(0, int(orig_word.start * original_sr) - guard_samples)
        orig_end = min(
            original_waveform.shape[1],
            int(orig_word.end * original_sr) + guard_samples,
        )
        resynth_start = max(
            0, int(resynth_word.start * original_sr) - guard_samples
        )
        resynth_end = min(
            resynth_waveform.shape[1],
            int(resynth_word.end * original_sr) + guard_samples,
        )

        resynth_segment = resynth_waveform[:, resynth_start:resynth_end]

        # Check if resynthesized word would overlap next word
        if idx + 1 < len(original_words):
            next_word_start = int(original_words[idx + 1].start * original_sr)
            max_end = next_word_start + int(max_overlap_ms * original_sr / 1000)
            available = max_end - orig_start
            if resynth_segment.shape[1] > available:
                logger.warning(
                    "Resynthesized word '%s' truncated: %d -> %d samples",
                    orig_word.word, resynth_segment.shape[1], available,
                )
                resynth_segment = resynth_segment[:, :available]

        # Overlap-add crossfade: snapshot the L samples about to be replaced
        # at each boundary, fade them complementarily, and SUM with the (faded)
        # resynth in the same region. Avoids the "dip through silence" you get
        # from sequential fade-out → fade-in across adjacent regions.
        fade_len = min(crossfade_samples, resynth_segment.shape[1] // 2)

        head_orig = None
        tail_orig = None
        if fade_len > 0:
            fade_in = _hann_crossfade(fade_len, device)
            fade_out = fade_in.flip(0)

            head_orig = output[:, orig_start : orig_start + fade_len].clone()
            tail_orig = output[:, max(0, orig_end - fade_len) : orig_end].clone()

            resynth_segment = resynth_segment.clone()
            resynth_segment[:, :fade_len] *= fade_in
            resynth_segment[:, -fade_len:] *= fade_out

        # Replace the word region
        orig_len = orig_end - orig_start
        resynth_len = resynth_segment.shape[1]

        if resynth_len == orig_len:
            output[:, orig_start:orig_end] = resynth_segment
            new_end = orig_end
        else:
            # Duration mismatch: reconstruct output with different length segment
            before = output[:, :orig_start]
            after = output[:, orig_end:]
            output = torch.cat([before, resynth_segment, after], dim=1)
            new_end = orig_start + resynth_len

        # Sum the (faded) original snapshots back into the crossfade regions.
        if fade_len > 0:
            output[:, orig_start : orig_start + fade_len] += head_orig * fade_out
            tail_len = tail_orig.shape[1]
            if tail_len > 0:
                output[:, new_end - tail_len : new_end] += tail_orig * fade_in[-tail_len:]

    return output
