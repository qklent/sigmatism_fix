"""Smoke tests for inherited Russian phonemizer module."""

from unittest.mock import MagicMock, patch


def test_russian_phonemizer_init():
    """RussianPhonemizer initializes without error (mocked backend)."""
    with patch("src.preprocessing.phonemizer.EspeakBackend"):
        from src.preprocessing.phonemizer import RussianPhonemizer

        p = RussianPhonemizer()
        assert p.language == "ru"
        assert p.preserve_punctuation is False


def test_preprocess_text():
    """preprocess_text strips whitespace and punctuation."""
    with patch("src.preprocessing.phonemizer.EspeakBackend"):
        from src.preprocessing.phonemizer import RussianPhonemizer

        p = RussianPhonemizer()
        assert p.preprocess_text("  привет мир  ") == "привет мир"
        assert p.preprocess_text("") == ""


def test_phonemize_empty_string():
    """phonemize returns empty string for empty input."""
    with patch("src.preprocessing.phonemizer.EspeakBackend"):
        from src.preprocessing.phonemizer import RussianPhonemizer

        p = RussianPhonemizer()
        result = p.phonemize("")
        assert result == ""


def test_phonemize_batch_empty():
    """phonemize_batch returns empty list for empty input."""
    with patch("src.preprocessing.phonemizer.EspeakBackend"):
        from src.preprocessing.phonemizer import RussianPhonemizer

        p = RussianPhonemizer()
        result = p.phonemize_batch([])
        assert result == []


def test_init_raises_when_espeak_backend_fails():
    """Constructor wraps backend exceptions in a RuntimeError."""
    import pytest

    with patch(
        "src.preprocessing.phonemizer.EspeakBackend",
        side_effect=OSError("espeak-ng missing"),
    ):
        from src.preprocessing.phonemizer import RussianPhonemizer

        with pytest.raises(RuntimeError, match="espeak"):
            RussianPhonemizer()


def test_phonemize_calls_phonemize_with_args():
    with (
        patch("src.preprocessing.phonemizer.EspeakBackend"),
        patch("src.preprocessing.phonemizer.phonemize", return_value="s a") as mock_p,
    ):
        from src.preprocessing.phonemizer import RussianPhonemizer

        p = RussianPhonemizer()
        out = p.phonemize("саша")
    assert out == "s a"
    mock_p.assert_called_once()


def test_phonemize_wraps_backend_runtime_errors():
    import pytest

    with (
        patch("src.preprocessing.phonemizer.EspeakBackend"),
        patch(
            "src.preprocessing.phonemizer.phonemize",
            side_effect=OSError("backend died"),
        ),
    ):
        from src.preprocessing.phonemizer import RussianPhonemizer

        with pytest.raises(RuntimeError, match="Phonemization failed"):
            RussianPhonemizer().phonemize("test")


def test_phonemize_batch_returns_list():
    with (
        patch("src.preprocessing.phonemizer.EspeakBackend"),
        patch(
            "src.preprocessing.phonemizer.phonemize",
            return_value=["s a", "ʂ a"],
        ),
    ):
        from src.preprocessing.phonemizer import RussianPhonemizer

        result = RussianPhonemizer().phonemize_batch(["саша", "шла"])
    assert result == ["s a", "ʂ a"]


def test_phonemize_batch_wraps_errors():
    import pytest

    with (
        patch("src.preprocessing.phonemizer.EspeakBackend"),
        patch(
            "src.preprocessing.phonemizer.phonemize",
            side_effect=OSError("batch failure"),
        ),
    ):
        from src.preprocessing.phonemizer import RussianPhonemizer

        with pytest.raises(RuntimeError, match="Batch phonemization failed"):
            RussianPhonemizer().phonemize_batch(["x", "y"])


def test_get_phoneme_list_splits_string_on_whitespace():
    with (
        patch("src.preprocessing.phonemizer.EspeakBackend"),
        patch(
            "src.preprocessing.phonemizer.phonemize",
            return_value="s a | sʲ t",
        ),
    ):
        from src.preprocessing.phonemizer import RussianPhonemizer

        out = RussianPhonemizer().get_phoneme_list("саша тест")
    assert out == ["s", "a", "sʲ", "t"]


def test_preprocess_text_with_preserve_punctuation_keeps_punct():
    with patch("src.preprocessing.phonemizer.EspeakBackend"):
        from src.preprocessing.phonemizer import RussianPhonemizer

        p = RussianPhonemizer(preserve_punctuation=True)
        # Punctuation kept because preserve_punctuation=True skips the regex strip
        assert "." in p.preprocess_text("привет, мир.")
