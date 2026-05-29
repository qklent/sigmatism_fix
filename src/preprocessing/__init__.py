"""Preprocessing modules for the inference pipeline.

Note: Some modules require optional [inference] dependencies (transformers,
phonemizer, sentencepiece). Import them directly when needed.
"""

from src.preprocessing.forced_aligner import MFAAligner, PhonemeAlignment
from src.preprocessing.hard_s_filter import filter_hard_s

__all__ = [
    "GigaAMTranscriber",
    "MFAAligner",
    "PhonemeAlignment",
    "RussianPhonemizer",
    "SileroVAD",
    "VadSegment",
    "filter_hard_s",
    "merge_segments",
    "slice_audio",
]

_VAD_NAMES = {"SileroVAD", "VadSegment", "merge_segments", "slice_audio"}


def __getattr__(name: str):
    if name == "GigaAMTranscriber":
        from src.preprocessing.asr import GigaAMTranscriber

        return GigaAMTranscriber
    if name == "RussianPhonemizer":
        from src.preprocessing.phonemizer import RussianPhonemizer

        return RussianPhonemizer
    if name in _VAD_NAMES:
        from src.preprocessing import vad

        return getattr(vad, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
