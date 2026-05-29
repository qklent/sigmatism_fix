"""
Forced alignment using Montreal Forced Aligner (MFA).
Aligns audio with phoneme sequences to get phoneme-level timestamps.

Adapted from speech_disorder_correction/mlm/data_preparation/src/forced_aligner.py.
Uses existing audio_adapter.py for audio I/O instead of AudioProcessor.
"""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhonemeAlignment:
    """Represents a single phoneme with its timing."""

    phoneme: str
    start: float
    end: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {"phoneme": self.phoneme, "start": self.start, "end": self.end}


@dataclass
class WordAlignment:
    """Represents a single word with timing and child phonemes from MFA alignment."""

    word: str
    start: float
    end: float
    phonemes: list[PhonemeAlignment]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "phonemes": [p.to_dict() for p in self.phonemes],
        }


class MFAAligner:
    """
    Wrapper for Montreal Forced Aligner.

    Requires MFA to be installed:
    conda install -c conda-forge montreal-forced-aligner
    """

    def __init__(
        self,
        acoustic_model: str = "russian_mfa",
        dictionary: str = "russian_mfa",
        temp_dir: Path | None = None,
        num_jobs: int = 4,
    ):
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.num_jobs = num_jobs

        if not self._check_mfa_installed():
            raise RuntimeError(
                "Montreal Forced Aligner (MFA) not found. "
                "Please install it: conda install -c conda-forge montreal-forced-aligner"
            )

        self._ensure_model_downloaded()

    def _check_mfa_installed(self) -> bool:
        return shutil.which("mfa") is not None

    def _ensure_model_downloaded(self):
        try:
            result = subprocess.run(
                ["mfa", "model", "list", "acoustic"],
                capture_output=True,
                text=True,
                check=False,
            )

            if self.acoustic_model not in result.stdout:
                logger.info("Downloading acoustic model: %s", self.acoustic_model)
                subprocess.run(
                    ["mfa", "model", "download", "acoustic", self.acoustic_model],
                    check=True,
                )

            result = subprocess.run(
                ["mfa", "model", "list", "dictionary"],
                capture_output=True,
                text=True,
                check=False,
            )

            if self.dictionary not in result.stdout:
                logger.info("Downloading dictionary: %s", self.dictionary)
                subprocess.run(
                    ["mfa", "model", "download", "dictionary", self.dictionary],
                    check=True,
                )

        except subprocess.CalledProcessError as e:
            logger.warning("Could not verify/download MFA models: %s", e)

    def prepare_mfa_input(self, audio_path: Path, text: str, output_dir: Path) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = "utterance"
        audio_out = output_dir / f"{base_name}.wav"
        text_out = output_dir / f"{base_name}.txt"

        if audio_path != audio_out:
            shutil.copy(audio_path, audio_out)

        text_out.write_text(text.strip(), encoding="utf-8")
        return audio_out, text_out

    def run_alignment(self, audio_path: Path, text: str, output_format: str = "json") -> Path:
        input_dir = self.temp_dir / "input"
        output_dir = self.temp_dir / "output"

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.prepare_mfa_input(audio_path, text, input_dir)

        cmd = [
            "mfa",
            "align",
            str(input_dir),
            self.dictionary,
            self.acoustic_model,
            str(output_dir),
            "--clean",
            "--output_format",
            output_format,
            "-j",
            str(self.num_jobs),
            "--use_mp",
            "--no_use_threading",
            "--single_speaker",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
            logger.debug("MFA stdout: %s", result.stdout[:500] if result.stdout else "")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"MFA alignment failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Return code: {e.returncode}\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            )

        if output_format == "json":
            output_file = output_dir / "utterance.json"
        else:
            output_file = output_dir / "utterance.TextGrid"

        if not output_file.exists():
            raise RuntimeError(f"MFA did not produce output file: {output_file}")

        return output_file

    def parse_mfa_json_full(
        self, json_path: Path
    ) -> tuple[list[PhonemeAlignment], list[WordAlignment]]:
        """Parse MFA JSON output returning both phoneme and word alignments.

        Phonemes are associated with parent words by timestamp containment.
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        phonemes: list[PhonemeAlignment] = []
        words: list[WordAlignment] = []

        # Parse phones tier
        if "tiers" in data and "phones" in data["tiers"]:
            phones_tier = data["tiers"]["phones"]
            if isinstance(phones_tier, dict) and "entries" in phones_tier:
                for entry in phones_tier["entries"]:
                    if len(entry) >= 3:
                        start_time, end_time, phoneme = entry[0], entry[1], entry[2]
                        if phoneme and phoneme not in ("", "sil", "sp", "spn"):
                            phonemes.append(
                                PhonemeAlignment(
                                    phoneme=phoneme,
                                    start=float(start_time),
                                    end=float(end_time),
                                )
                            )

        # Parse words tier
        if "tiers" in data and "words" in data["tiers"]:
            words_tier = data["tiers"]["words"]
            if isinstance(words_tier, dict) and "entries" in words_tier:
                for entry in words_tier["entries"]:
                    if len(entry) >= 3:
                        start_time, end_time, word_text = entry[0], entry[1], entry[2]
                        if word_text and word_text not in ("", "<eps>"):
                            w_start = float(start_time)
                            w_end = float(end_time)
                            # Associate phonemes by timestamp containment
                            child_phonemes = [
                                p for p in phonemes
                                if p.start >= w_start - 1e-6 and p.end <= w_end + 1e-6
                            ]
                            words.append(
                                WordAlignment(
                                    word=word_text,
                                    start=w_start,
                                    end=w_end,
                                    phonemes=child_phonemes,
                                )
                            )

        return phonemes, words

    def parse_mfa_json(self, json_path: Path) -> list[PhonemeAlignment]:
        """Parse MFA JSON output returning only phoneme alignments (backward compat)."""
        phonemes, _ = self.parse_mfa_json_full(json_path)
        return phonemes

    def align(self, audio: np.ndarray, text: str, sample_rate: int = 16000) -> list[PhonemeAlignment]:
        import soundfile as sf

        temp_audio = self.temp_dir / "temp_audio.wav"
        sf.write(temp_audio, audio, sample_rate)

        output_json = self.run_alignment(temp_audio, text, output_format="json")
        alignments = self.parse_mfa_json(output_json)

        temp_audio.unlink(missing_ok=True)
        return alignments

    def align_full(
        self, audio: np.ndarray, text: str, sample_rate: int = 16000
    ) -> tuple[list[PhonemeAlignment], list[WordAlignment]]:
        """Run alignment returning both phoneme and word alignments."""
        import soundfile as sf

        temp_audio = self.temp_dir / "temp_audio.wav"
        sf.write(temp_audio, audio, sample_rate)

        output_json = self.run_alignment(temp_audio, text, output_format="json")
        result = self.parse_mfa_json_full(output_json)

        temp_audio.unlink(missing_ok=True)
        return result

    def align_batch(
        self,
        items: list[tuple[str, np.ndarray, str]],
        sample_rate: int = 16000,
    ) -> dict[str, tuple[list[PhonemeAlignment], list[WordAlignment]]]:
        """Align many utterances in a single MFA invocation.

        items: list of (utt_id, audio, text). utt_id must be filesystem-safe
        and unique within the batch. Returns a dict keyed by utt_id; ids whose
        alignment produced no JSON are omitted (caller counts misses).

        Amortizes MFA model load over the whole batch — ~60s fixed cost
        instead of once per utterance.
        """
        import soundfile as sf

        input_dir = self.temp_dir / "batch_input"
        output_dir = self.temp_dir / "batch_output"
        if input_dir.exists():
            shutil.rmtree(input_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        for utt_id, audio, text in items:
            wav_path = input_dir / f"{utt_id}.wav"
            txt_path = input_dir / f"{utt_id}.txt"
            sf.write(wav_path, audio, sample_rate)
            txt_path.write_text(text.strip(), encoding="utf-8")

        cmd = [
            "mfa",
            "align",
            str(input_dir),
            self.dictionary,
            self.acoustic_model,
            str(output_dir),
            "--clean",
            "--output_format",
            "json",
            "-j",
            str(self.num_jobs),
            "--use_mp",
            "--no_use_threading",
            "--single_speaker",
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"MFA batch alignment failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Return code: {e.returncode}\n"
                f"Stderr: {e.stderr}"
            )

        results: dict[str, tuple[list[PhonemeAlignment], list[WordAlignment]]] = {}
        for utt_id, _a, _t in items:
            out_json = output_dir / f"{utt_id}.json"
            if out_json.exists():
                results[utt_id] = self.parse_mfa_json_full(out_json)
        return results

    def align_from_file(self, audio_path: str | Path, text: str) -> list[PhonemeAlignment]:
        """Run alignment directly from an audio file path."""
        output_json = self.run_alignment(Path(audio_path), text, output_format="json")
        return self.parse_mfa_json(output_json)

    def align_from_file_full(
        self, audio_path: str | Path, text: str
    ) -> tuple[list[PhonemeAlignment], list[WordAlignment]]:
        """Run alignment from file returning both phoneme and word alignments."""
        output_json = self.run_alignment(Path(audio_path), text, output_format="json")
        return self.parse_mfa_json_full(output_json)

    def cleanup(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
