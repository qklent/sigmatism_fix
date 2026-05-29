"""Unit tests for the GigaAM RNN-T word aligner."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.preprocessing.forced_aligner import WordAlignment
from src.preprocessing.gigaam_aligner import FRAME_STRIDE_S, GigaAMAligner


def _build_transcriber(timed_tokens, pieces_for_id):
    """Construct a mock GigaAMTranscriber with canned RNN-T emissions.

    ``pieces_for_id`` maps token_id (int) -> SentencePiece piece string.
    """
    transcriber = MagicMock()
    transcriber.transcribe_with_timings.return_value = ("dummy text", list(timed_tokens))
    transcriber.model.model.decoding.tokenizer.model.id_to_piece.side_effect = (
        lambda i: pieces_for_id[int(i)]
    )
    return transcriber


def test_align_from_file_full_basic_word_grouping():
    """Two emitted words map cleanly to two transcript words."""
    pieces = {1: "▁са", 2: "ша", 3: "▁шла"}
    timed_tokens = [(5, 1), (7, 2), (15, 3)]  # frames 5, 7, 15
    transcriber = _build_transcriber(timed_tokens, pieces)

    aligner = GigaAMAligner(transcriber=transcriber, late_bias_ms=0.0)
    phonemes, words = aligner.align_from_file_full("/fake/audio.wav", "Саша шла")

    assert phonemes == []
    assert len(words) == 2
    assert words[0].word == "Саша"
    assert words[1].word == "шла"
    # First word starts at frame 5
    assert words[0].start == 5 * FRAME_STRIDE_S
    # Last word's end is one past its last token's frame
    assert words[1].end == (15 + 1) * FRAME_STRIDE_S


def test_align_from_file_full_late_bias_shifts_timestamps():
    """A non-zero late_bias_ms shifts word endpoints earlier (and clamps at 0)."""
    pieces = {1: "▁hi"}
    timed_tokens = [(10, 1)]
    transcriber = _build_transcriber(timed_tokens, pieces)

    aligner = GigaAMAligner(transcriber=transcriber, late_bias_ms=80.0)
    _, words = aligner.align_from_file_full("/fake/a.wav", "hi")

    assert len(words) == 1
    # start: max(0, 10 * 0.04 - 0.08) = 0.32; end: (10+1)*0.04 - 0.08 = 0.36
    assert abs(words[0].start - 0.32) < 1e-9
    assert abs(words[0].end - 0.36) < 1e-9


def test_align_from_file_full_clamps_negative_start_to_zero():
    """When late_bias exceeds the start time, start is clamped to 0 and end >= start."""
    pieces = {1: "▁x"}
    timed_tokens = [(0, 1)]
    transcriber = _build_transcriber(timed_tokens, pieces)

    aligner = GigaAMAligner(transcriber=transcriber, late_bias_ms=500.0)
    _, words = aligner.align_from_file_full("/fake/a.wav", "x")

    assert words[0].start == 0.0
    assert words[0].end >= words[0].start


def test_align_from_file_full_count_mismatch_uses_positional(caplog):
    """If emitted-word count != transcript-word count, take min and log a warning."""
    pieces = {1: "▁a", 2: "▁b", 3: "▁c"}
    timed_tokens = [(0, 1), (5, 2), (10, 3)]
    transcriber = _build_transcriber(timed_tokens, pieces)

    aligner = GigaAMAligner(transcriber=transcriber, late_bias_ms=0.0)
    with caplog.at_level("WARNING"):
        _, words = aligner.align_from_file_full("/fake/a.wav", "only two")

    # Two transcript words → two aligned words; the third emitted word is dropped.
    assert len(words) == 2
    assert [w.word for w in words] == ["only", "two"]
    assert any("mismatch" in rec.message for rec in caplog.records)


def test_align_from_file_full_leading_non_underscore_piece():
    """A token without ▁ prefix at the start of the sequence still seeds a word."""
    pieces = {1: "no_prefix", 2: "▁next"}
    timed_tokens = [(3, 1), (8, 2)]
    transcriber = _build_transcriber(timed_tokens, pieces)

    aligner = GigaAMAligner(transcriber=transcriber, late_bias_ms=0.0)
    _, words = aligner.align_from_file_full("/fake/a.wav", "first next")

    assert len(words) == 2
    # First word started at frame 3 (leading non-▁ branch in gigaam_aligner)
    assert words[0].start == 3 * FRAME_STRIDE_S
    assert words[0].end == 8 * FRAME_STRIDE_S


def test_align_returns_word_alignment_dataclass():
    pieces = {1: "▁word"}
    transcriber = _build_transcriber([(0, 1)], pieces)
    aligner = GigaAMAligner(transcriber=transcriber, late_bias_ms=0.0)
    _, words = aligner.align_from_file_full("/fake/a.wav", "word")
    assert isinstance(words[0], WordAlignment)
    assert words[0].phonemes == []  # GigaAM doesn't emit phonemes
