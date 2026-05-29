"""Unit tests for hard-S phoneme filter."""

from unittest.mock import MagicMock

from src.preprocessing.forced_aligner import PhonemeAlignment, WordAlignment
from src.preprocessing.hard_s_filter import (
    HARD_S_PHONEME,
    filter_hard_s,
    filter_hard_s_words,
    filter_hard_s_words_by_text,
)


def test_filter_hard_s_with_mixed_phonemes():
    alignments = [
        PhonemeAlignment(phoneme="a", start=0.0, end=0.1),
        PhonemeAlignment(phoneme=HARD_S_PHONEME, start=0.5, end=0.6),
        PhonemeAlignment(phoneme="pʲ", start=0.7, end=0.8),
        PhonemeAlignment(phoneme=HARD_S_PHONEME, start=1.0, end=1.15),
    ]
    result = filter_hard_s(alignments)
    assert len(result) == 2
    assert result[0] == {"start": 0.5, "end": 0.6}
    assert result[1] == {"start": 1.0, "end": 1.15}


def test_filter_hard_s_empty_list():
    assert filter_hard_s([]) == []


def test_filter_hard_s_no_hard_s():
    alignments = [
        PhonemeAlignment(phoneme="a", start=0.0, end=0.1),
        PhonemeAlignment(phoneme="b", start=0.1, end=0.2),
    ]
    assert filter_hard_s(alignments) == []


def test_filter_hard_s_output_format():
    alignments = [PhonemeAlignment(phoneme=HARD_S_PHONEME, start=2.5, end=2.7)]
    result = filter_hard_s(alignments)
    assert len(result) == 1
    seg = result[0]
    assert isinstance(seg, dict)
    assert "start" in seg and "end" in seg
    assert isinstance(seg["start"], float)
    assert isinstance(seg["end"], float)


def test_filter_hard_s_all_hard_s():
    alignments = [
        PhonemeAlignment(phoneme=HARD_S_PHONEME, start=0.0, end=0.1),
        PhonemeAlignment(phoneme=HARD_S_PHONEME, start=0.2, end=0.3),
    ]
    result = filter_hard_s(alignments)
    assert len(result) == 2


# -- filter_hard_s_words ------------------------------------------------------


def test_filter_hard_s_words_returns_words_with_hard_s_phoneme():
    p_hard = PhonemeAlignment(phoneme=HARD_S_PHONEME, start=0.1, end=0.2)
    p_other = PhonemeAlignment(phoneme="a", start=0.2, end=0.3)
    w_hard = WordAlignment(word="саша", start=0.1, end=0.3, phonemes=[p_hard, p_other])
    w_clean = WordAlignment(
        word="чисто", start=0.4, end=0.6,
        phonemes=[PhonemeAlignment(phoneme="t", start=0.4, end=0.5)],
    )
    out = filter_hard_s_words([w_hard, w_clean], [p_hard, p_other])
    assert out == [w_hard]


def test_filter_hard_s_words_excludes_words_without_matching_timing():
    """A hard-S phoneme that lives outside this word's phoneme list is ignored."""
    p_hard = PhonemeAlignment(phoneme=HARD_S_PHONEME, start=0.9, end=1.0)
    w = WordAlignment(
        word="без", start=0.0, end=0.3,
        phonemes=[PhonemeAlignment(phoneme="b", start=0.0, end=0.3)],
    )
    assert filter_hard_s_words([w], [p_hard]) == []


# -- filter_hard_s_words_by_text ---------------------------------------------


def test_filter_hard_s_words_by_text_keeps_hard_s_words():
    phon = MagicMock()
    phon.get_phoneme_list.side_effect = lambda w: {
        "саша": ["s", "a", "ʂ", "a"],
        "чисто": ["tɕ", "i", "sʲ", "t", "a"],
    }[w]
    words = [
        WordAlignment(word="саша", start=0.0, end=0.5, phonemes=[]),
        WordAlignment(word="чисто", start=0.5, end=1.0, phonemes=[]),
    ]
    out = filter_hard_s_words_by_text(words, phonemizer=phon)
    assert [w.word for w in out] == ["саша"]


def test_filter_hard_s_words_by_text_swallows_phonemizer_exceptions():
    phon = MagicMock()
    phon.get_phoneme_list.side_effect = RuntimeError("espeak broke")
    out = filter_hard_s_words_by_text(
        [WordAlignment(word="x", start=0.0, end=0.1, phonemes=[])], phonemizer=phon
    )
    assert out == []


def test_filter_hard_s_words_by_text_default_phonemizer_path():
    """When phonemizer=None the function constructs a RussianPhonemizer."""
    from unittest.mock import patch

    with patch(
        "src.preprocessing.phonemizer.RussianPhonemizer"
    ) as mock_cls:
        instance = MagicMock()
        instance.get_phoneme_list.return_value = ["s"]
        mock_cls.return_value = instance
        out = filter_hard_s_words_by_text(
            [WordAlignment(word="word", start=0.0, end=0.1, phonemes=[])]
        )
    assert len(out) == 1
    mock_cls.assert_called_once()
