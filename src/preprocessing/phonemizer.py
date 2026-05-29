"""
Text to phoneme conversion for Russian language.
Uses phonemizer library with espeak-ng backend.

Adapted from speech_disorder_correction/mlm/data_preparation/src/phonemizer.py.
"""

import re

from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator


class RussianPhonemizer:
    """Convert Russian text to phonemes using espeak-ng."""

    def __init__(self, language: str = "ru", preserve_punctuation: bool = False):
        self.language = language
        self.preserve_punctuation = preserve_punctuation
        self.separator = Separator(phone=" ", word=" | ", syllable="")

        try:
            self.backend = EspeakBackend(
                language=language,
                preserve_punctuation=preserve_punctuation,
                with_stress=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize espeak backend. "
                f"Make sure espeak-ng is installed: {e}"
            )

    def preprocess_text(self, text: str) -> str:
        text = " ".join(text.split())

        if not self.preserve_punctuation:
            text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)

        return text.strip()

    def phonemize(self, text: str) -> str:
        text = self.preprocess_text(text)

        if not text:
            return ""

        try:
            phonemes = phonemize(
                text,
                language=self.language,
                backend="espeak",
                separator=self.separator,
                strip=True,
                preserve_punctuation=self.preserve_punctuation,
                with_stress=True,
            )
            return phonemes
        except Exception as e:
            raise RuntimeError(f"Phonemization failed for text '{text}': {e}")

    def phonemize_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        preprocessed = [self.preprocess_text(text) for text in texts]

        try:
            return phonemize(
                preprocessed,
                language=self.language,
                backend="espeak",
                separator=self.separator,
                strip=True,
                preserve_punctuation=self.preserve_punctuation,
                with_stress=True,
                njobs=4,
            )
        except Exception as e:
            raise RuntimeError(f"Batch phonemization failed: {e}")

    def get_phoneme_list(self, text: str) -> list[str]:
        phoneme_string = self.phonemize(text)
        phoneme_string = phoneme_string.replace(" | ", " ")
        return [p for p in phoneme_string.split() if p]
