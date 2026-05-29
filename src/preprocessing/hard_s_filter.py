"""Filter phoneme alignments for hard-S (s̪) phonemes."""

from __future__ import annotations

from src.preprocessing.forced_aligner import PhonemeAlignment, WordAlignment

HARD_S_PHONEME = "s̪"
ESPEAK_HARD_S = "s"  # espeak distinguishes 's' (hard) from 'sʲ' (palatalized)


def filter_hard_s(alignments: list[PhonemeAlignment]) -> list[dict[str, float]]:
    """Filter alignments for hard-S phonemes and return timestamp dicts.

    Args:
        alignments: List of phoneme alignments from MFA.

    Returns:
        List of {"start": float, "end": float} dicts for each hard-S segment.
    """
    return [
        {"start": a.start, "end": a.end}
        for a in alignments
        if a.phoneme == HARD_S_PHONEME
    ]


def filter_hard_s_words(
    word_alignments: list[WordAlignment],
    phoneme_alignments: list[PhonemeAlignment],
) -> list[WordAlignment]:
    """Return words that contain at least one hard-S phoneme.

    Uses the existing filter_hard_s() to identify hard-S phonemes, then maps
    them back to parent words via timestamp matching.
    """
    hard_s_segments = filter_hard_s(phoneme_alignments)
    hard_s_times = {(seg["start"], seg["end"]) for seg in hard_s_segments}

    return [
        word for word in word_alignments
        if any((p.start, p.end) in hard_s_times for p in word.phonemes)
    ]


def filter_hard_s_words_by_text(
    word_alignments: list[WordAlignment],
    phonemizer=None,
) -> list[WordAlignment]:
    """Return words whose espeak phonemization contains a non-palatalized 's'.

    Used when alignments come from an aligner that doesn't produce phonemes
    (e.g. GigaAMAligner). Espeak emits 's' for hard /s/ and 'sʲ' for soft —
    splitting on whitespace and matching exactly 's' is the correct check.
    """
    from src.preprocessing.phonemizer import RussianPhonemizer

    if phonemizer is None:
        phonemizer = RussianPhonemizer()

    out: list[WordAlignment] = []
    for w in word_alignments:
        try:
            phones = phonemizer.get_phoneme_list(w.word)
        except Exception:
            continue
        if any(p == ESPEAK_HARD_S for p in phones):
            out.append(w)
    return out
