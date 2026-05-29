"""Unit tests for word-level splicing (T015)."""

import torch

from src.inference.word_splice import splice_words
from src.preprocessing.forced_aligner import PhonemeAlignment, WordAlignment


def _make_word(word: str, start: float, end: float) -> WordAlignment:
    return WordAlignment(word=word, start=start, end=end, phonemes=[])


class TestSpliceWords:
    def test_non_target_regions_unchanged(self):
        """Regions outside target words should remain identical."""
        sr = 16000
        orig = torch.randn(1, sr * 2)  # 2 seconds
        resynth = torch.randn(1, sr * 2)

        orig_words = [
            _make_word("а", 0.0, 0.5),
            _make_word("б", 0.6, 1.0),
            _make_word("в", 1.2, 1.8),
        ]
        resynth_words = [
            _make_word("а", 0.0, 0.5),
            _make_word("б", 0.5, 0.9),
            _make_word("в", 1.0, 1.6),
        ]

        result = splice_words(
            orig, sr, orig_words, resynth, sr, resynth_words,
            target_word_indices=[1], crossfade_ms=0,
        )

        # Region before word 1 (samples 0 to 0.6*sr) should be unchanged
        # (word 0 is not targeted)
        assert torch.equal(result[:, :int(0.5 * sr)], orig[:, :int(0.5 * sr)])

    def test_crossfade_applied(self):
        """With crossfade > 0, boundaries should not have hard transitions."""
        sr = 16000
        orig = torch.ones(1, sr * 2)
        resynth = torch.zeros(1, sr * 2)

        orig_words = [_make_word("а", 0.5, 1.0)]
        resynth_words = [_make_word("а", 0.5, 1.0)]

        result = splice_words(
            orig, sr, orig_words, resynth, sr, resynth_words,
            target_word_indices=[0], crossfade_ms=10.0,
        )

        # The replaced region should contain zeros (from resynth) modulated by crossfade
        mid = int(0.75 * sr)
        assert result[0, mid].abs() < 0.01  # middle of replaced word is ~0

    def test_sample_rate_resampling(self):
        """Resynthesized audio at different SR should be resampled."""
        orig_sr = 16000
        resynth_sr = 24000
        orig = torch.ones(1, orig_sr * 2)
        resynth = torch.full((1, resynth_sr * 2), 0.5)

        orig_words = [_make_word("а", 0.5, 1.0)]
        resynth_words = [_make_word("а", 0.5, 1.0)]

        result = splice_words(
            orig, orig_sr, orig_words, resynth, resynth_sr, resynth_words,
            target_word_indices=[0], crossfade_ms=0,
        )

        # Output should be at original SR length (approximately)
        assert result.shape[1] == orig.shape[1]

    def test_shorter_replacement(self):
        """Shorter resynthesized word should produce shorter output."""
        sr = 16000
        orig = torch.ones(1, sr * 2)
        resynth = torch.zeros(1, sr * 2)

        # Original word is 0.5s, resynthesized is 0.3s
        orig_words = [_make_word("а", 0.5, 1.0)]
        resynth_words = [_make_word("а", 0.5, 0.8)]

        result = splice_words(
            orig, sr, orig_words, resynth, sr, resynth_words,
            target_word_indices=[0], crossfade_ms=0,
        )

        # Output should be shorter by the duration difference
        expected_diff = int(0.2 * sr)  # 0.5s - 0.3s
        assert abs(result.shape[1] - (orig.shape[1] - expected_diff)) < 10

    def test_longer_replacement(self):
        """Longer resynthesized word should produce longer output."""
        sr = 16000
        orig = torch.ones(1, sr * 2)
        resynth = torch.zeros(1, sr * 2)

        # Original word is 0.3s, resynthesized is 0.5s
        orig_words = [_make_word("а", 0.5, 0.8), _make_word("б", 1.2, 1.5)]
        resynth_words = [_make_word("а", 0.5, 1.0), _make_word("б", 1.2, 1.5)]

        result = splice_words(
            orig, sr, orig_words, resynth, sr, resynth_words,
            target_word_indices=[0], crossfade_ms=0,
        )

        # Output should be longer
        expected_diff = int(0.2 * sr)
        assert abs(result.shape[1] - (orig.shape[1] + expected_diff)) < 10

    def test_1d_input_handled(self):
        """1D input tensors should be handled correctly."""
        sr = 16000
        orig = torch.ones(sr * 2)
        resynth = torch.zeros(sr * 2)

        orig_words = [_make_word("а", 0.5, 1.0)]
        resynth_words = [_make_word("а", 0.5, 1.0)]

        result = splice_words(
            orig, sr, orig_words, resynth, sr, resynth_words,
            target_word_indices=[0], crossfade_ms=0,
        )

        assert result.dim() == 2
        assert result.shape[0] == 1

    def test_out_of_range_index_skipped(self):
        """Out-of-range word indices should be skipped with a warning."""
        sr = 16000
        orig = torch.ones(1, sr)
        resynth = torch.zeros(1, sr)

        orig_words = [_make_word("а", 0.0, 0.5)]
        resynth_words = [_make_word("а", 0.0, 0.5)]

        result = splice_words(
            orig, sr, orig_words, resynth, sr, resynth_words,
            target_word_indices=[5], crossfade_ms=0,
        )

        # Should return original unchanged
        assert torch.equal(result, orig)
