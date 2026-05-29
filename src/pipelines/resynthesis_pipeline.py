"""Resynthesis pipeline for hard-S correction via OmniVoice.

Orchestrates: MFA word-align -> filter hard-S words -> OmniVoice resynthesize
-> MFA align resynthesized -> word-level splice -> output.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ContextManager, Protocol, runtime_checkable

import torch
import torchaudio

from src.config.schema import OmniVoiceConfig, ResynthesisConfig
from src.data.audio_adapter import save_waveform
from src.inference.omnivoice_adapter import OmniVoiceAdapter
from src.inference.word_splice import splice_words
from src.preprocessing.forced_aligner import MFAAligner
from src.preprocessing.hard_s_filter import (
    filter_hard_s_words,
    filter_hard_s_words_by_text,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class StageProbe(Protocol):
    """Hook surface used by the stress-test harness (feature 019).

    Production callers pass ``probe=None``; ``ResynthesisPipeline`` substitutes
    ``_NullProbe`` so every probe call collapses to a no-op context manager.
    """

    def stage(self, name: str, *, gpu: bool) -> ContextManager: ...
    def end_to_end(self) -> ContextManager: ...


class _NullProbe:
    """Zero-overhead probe used when no stress harness is attached."""

    def stage(self, name: str, *, gpu: bool) -> ContextManager:  # noqa: ARG002
        return contextlib.nullcontext()

    def end_to_end(self) -> ContextManager:
        return contextlib.nullcontext()


class ResynthesisStatus(str, enum.Enum):
    success = "success"
    no_segments_found = "no_segments_found"
    alignment_failed = "alignment_failed"
    resynthesis_failed = "resynthesis_failed"
    error = "error"


@dataclass
class CorrectedWord:
    """Detail record for a corrected word."""

    word: str
    original_start: float
    original_end: float
    resynth_start: float
    resynth_end: float
    duration_delta_ms: float


@dataclass
class ResynthesisResult:
    """Output of the resynthesis pipeline for a single audio file."""

    input_path: str
    output_path: str
    transcript: str = ""
    ref_transcript: str | None = None
    words_detected: int = 0
    words_corrected: int = 0
    word_details: list[CorrectedWord] = field(default_factory=list)
    status: ResynthesisStatus = ResynthesisStatus.success
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "status": self.status.value,
            "ref_transcript": self.ref_transcript,
            "words_detected": self.words_detected,
            "words_corrected": self.words_corrected,
            "word_details": [
                {
                    "word": w.word,
                    "original_start": w.original_start,
                    "original_end": w.original_end,
                    "duration_delta_ms": w.duration_delta_ms,
                }
                for w in self.word_details
            ],
            "error_message": self.error_message,
        }


@dataclass
class BatchResynthesisReport:
    """Aggregate report for batch processing."""

    total_files: int = 0
    successful: int = 0
    no_segments: int = 0
    failed: int = 0
    results: list[ResynthesisResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "successful": self.successful,
            "no_segments": self.no_segments,
            "failed": self.failed,
            "failure_rate": self.failed / max(self.total_files, 1),
        }


class ResynthesisPipeline:
    """Orchestrates OmniVoice resynthesis with word-level splicing."""

    def __init__(
        self,
        omnivoice_cfg: OmniVoiceConfig,
        resynthesis_cfg: ResynthesisConfig,
        device: str | None = None,
        probe: StageProbe | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.omnivoice_cfg = omnivoice_cfg
        self.resynthesis_cfg = resynthesis_cfg
        self._probe: StageProbe = probe if probe is not None else _NullProbe()

        self._aligner_kind = resynthesis_cfg.aligner
        if self._aligner_kind == "mfa":
            logger.info("Initializing MFA aligner...")
            self.aligner = MFAAligner()
        else:
            logger.info("Using GigaAM RNN-T aligner (lazy-loaded with ASR)")
            self.aligner = None  # built lazily, shares the GigaAM model with ASR

        logger.info("Initializing OmniVoice adapter...")
        self.tts = OmniVoiceAdapter(omnivoice_cfg, device=device)

        self._asr = None

    def _get_asr(self):
        """Lazy-load GigaAM transcriber (only when transcript is not provided)."""
        if self._asr is None:
            from src.preprocessing.asr import GigaAMTranscriber

            logger.info("Initializing GigaAM ASR for auto-transcription...")
            self._asr = GigaAMTranscriber(device=self.device)
        return self._asr

    def _get_gigaam_aligner(self):
        """Lazy-init GigaAMAligner sharing the ASR model."""
        if self.aligner is None:
            from src.preprocessing.gigaam_aligner import GigaAMAligner

            self.aligner = GigaAMAligner(transcriber=self._get_asr())
        return self.aligner

    def _align(self, audio_path: str | Path, transcript: str):
        """Dispatch alignment to the configured backend.

        Returns ``(phonemes, words)`` — phonemes will be empty for the GigaAM
        backend.
        """
        if self._aligner_kind == "mfa":
            return self.aligner.align_from_file_full(audio_path, transcript)
        return self._get_gigaam_aligner().align_from_file_full(audio_path, transcript)

    def process_file(
        self,
        audio_path: str | Path,
        transcript: str | None,
        output_path: str | Path,
        ref_audio_path: str | Path | None = None,
        ref_text: str | None = None,
        save_raw_tts: bool = False,
    ) -> ResynthesisResult:
        """Process a single audio file through the resynthesis pipeline.

        Args:
            audio_path: Path to input audio.
            transcript: Text transcript. If None, GigaAM ASR is used automatically.
            output_path: Path to save corrected audio.
            ref_audio_path: Optional clean reference audio for voice cloning.
                When None, the disordered input is used as its own reference.
            ref_text: Optional transcript for ``ref_audio_path``. If omitted and
                ``resynthesis.provide_ref_text`` is true, GigaAM ASR is used.
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-transcribe if no transcript provided
        if transcript is None:
            logger.info("No transcript provided — running GigaAM ASR on %s", audio_path.name)
            try:
                with self._probe.stage("asr", gpu=True):
                    transcript = self._get_asr().transcribe(str(audio_path))
                logger.info("ASR result: '%s'", transcript)
            except Exception as e:
                return ResynthesisResult(
                    input_path=str(audio_path),
                    output_path=str(output_path),
                    status=ResynthesisStatus.error,
                    error_message=f"ASR transcription failed: {e}",
                )

        result = ResynthesisResult(
            input_path=str(audio_path),
            output_path=str(output_path),
            transcript=transcript,
        )

        # Step 1: Load original audio at native SR
        try:
            with self._probe.stage("load", gpu=False):
                waveform, orig_sr = torchaudio.load(str(audio_path))
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                duration = waveform.shape[1] / orig_sr
                if duration < 1.0:
                    logger.warning("Audio is very short (%.2fs): %s", duration, audio_path.name)
        except Exception as e:
            result.status = ResynthesisStatus.error
            result.error_message = f"Failed to load audio: {e}"
            return result

        # Step 2: Align original -> phonemes + words
        try:
            logger.info("Aligning original audio with %s...", self._aligner_kind)
            t0 = time.perf_counter()
            with self._probe.stage("align_orig", gpu=True):
                phonemes, orig_words = self._align(audio_path, transcript)
            logger.info(
                "Original alignment took %.3fs (aligner=%s, words=%d, phonemes=%d)",
                time.perf_counter() - t0,
                self._aligner_kind,
                len(orig_words),
                len(phonemes),
            )
        except Exception as e:
            logger.error("Alignment failed for %s: %s", audio_path.name, e)
            result.status = ResynthesisStatus.alignment_failed
            result.error_message = f"Original alignment failed: {e}"
            return result

        # Step 3: Filter hard-S words. With MFA we have phonemes; with GigaAM
        # we fall back to text-based detection via the espeak phonemizer.
        with self._probe.stage("hard_s_filter", gpu=False):
            if self._aligner_kind == "mfa":
                hard_s_words = filter_hard_s_words(orig_words, phonemes)
            else:
                hard_s_words = filter_hard_s_words_by_text(orig_words)
        result.words_detected = len(hard_s_words)

        if not hard_s_words:
            logger.info("No hard-S words found in %s. Copying original.", audio_path.name)
            shutil.copy2(audio_path, output_path)
            result.status = ResynthesisStatus.no_segments_found
            return result

        logger.info("Found %d hard-S words: %s", len(hard_s_words),
                     [w.word for w in hard_s_words])

        # Step 4: Synthesize full sentence via OmniVoice
        try:
            logger.info("Resynthesizing via OmniVoice...")
            if ref_audio_path is not None:
                tts_ref_audio = str(ref_audio_path)
                if self.resynthesis_cfg.provide_ref_text:
                    if ref_text is not None:
                        tts_ref_text = ref_text
                    else:
                        logger.info("No ref_text provided — running GigaAM ASR on reference %s", tts_ref_audio)
                        tts_ref_text = self._get_asr().transcribe(tts_ref_audio)
                        logger.info("Reference ASR result: '%s'", tts_ref_text)
                else:
                    tts_ref_text = None
                result.ref_transcript = tts_ref_text
            else:
                tts_ref_audio = str(audio_path)
                tts_ref_text = transcript if self.resynthesis_cfg.provide_ref_text else None
            with self._probe.stage("tts", gpu=True):
                resynth_waveform, resynth_sr = self.tts.synthesize(
                    text=transcript,
                    ref_audio_path=tts_ref_audio,
                    ref_text=tts_ref_text,
                )

            if save_raw_tts:
                raw_path = output_path.with_name(
                    output_path.stem + ".raw_tts.wav"
                )
                save_waveform(
                    resynth_waveform.squeeze(0).cpu(), raw_path, resynth_sr
                )
                logger.info("Saved raw OmniVoice output to %s", raw_path)
        except Exception as e:
            logger.exception("OmniVoice synthesis failed")
            result.status = ResynthesisStatus.resynthesis_failed
            result.error_message = f"Resynthesis failed: {e}"
            return result

        # Step 5: Align resynthesized audio -> words
        try:
            logger.info("Aligning resynthesized audio with %s...", self._aligner_kind)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            save_waveform(resynth_waveform.squeeze(0).cpu(), tmp_path, resynth_sr)
            t0 = time.perf_counter()
            with self._probe.stage("align_resynth", gpu=True):
                _, resynth_words = self._align(tmp_path, transcript)
            logger.info(
                "Resynth alignment took %.3fs (aligner=%s, words=%d)",
                time.perf_counter() - t0,
                self._aligner_kind,
                len(resynth_words),
            )
            tmp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error("Resynthesized alignment failed: %s", e)
            tmp_path.unlink(missing_ok=True)
            result.status = ResynthesisStatus.alignment_failed
            result.error_message = f"Resynthesized alignment failed: {e}"
            return result

        # Step 6: Match words and determine target indices
        target_indices = self._match_target_indices(orig_words, resynth_words, hard_s_words)

        if not target_indices:
            logger.warning("Could not match any hard-S words. Copying original.")
            shutil.copy2(audio_path, output_path)
            result.status = ResynthesisStatus.no_segments_found
            return result

        # Step 7: Splice corrected words into original
        try:
            with self._probe.stage("splice", gpu=False):
                output_waveform = splice_words(
                    original_waveform=waveform,
                    original_sr=orig_sr,
                    original_words=orig_words,
                    resynth_waveform=resynth_waveform.cpu(),
                    resynth_sr=resynth_sr,
                    resynth_words=resynth_words,
                    target_word_indices=target_indices,
                    crossfade_ms=self.resynthesis_cfg.crossfade_ms,
                    max_overlap_ms=self.resynthesis_cfg.max_overlap_ms,
                    guard_ms=self.resynthesis_cfg.guard_ms,
                )
        except Exception as e:
            result.status = ResynthesisStatus.error
            result.error_message = f"Splicing failed: {e}"
            return result

        # Step 8: Save output
        with self._probe.stage("save", gpu=False):
            out_sr = self.resynthesis_cfg.output_sample_rate or orig_sr
            if out_sr != orig_sr:
                output_waveform = torchaudio.functional.resample(output_waveform, orig_sr, out_sr)
            save_waveform(output_waveform.squeeze(0), output_path, out_sr)

        # Step 9: Build word details
        for idx in target_indices:
            orig_w = orig_words[idx]
            resynth_w = resynth_words[idx]
            orig_dur = (orig_w.end - orig_w.start) * 1000
            resynth_dur = (resynth_w.end - resynth_w.start) * 1000
            result.word_details.append(CorrectedWord(
                word=orig_w.word,
                original_start=orig_w.start,
                original_end=orig_w.end,
                resynth_start=resynth_w.start,
                resynth_end=resynth_w.end,
                duration_delta_ms=resynth_dur - orig_dur,
            ))

        result.words_corrected = len(target_indices)
        result.status = ResynthesisStatus.success
        logger.info("Corrected %d words in %s -> %s",
                     result.words_corrected, audio_path.name, output_path)
        return result

    def _match_target_indices(self, orig_words, resynth_words, hard_s_words):
        """Match hard-S words to their indices in the original word list.

        Primary strategy: index matching (same transcript -> same word order).
        Fallback: text matching by word string if word counts differ.
        """
        # Build set of hard-S word indices in original
        hard_s_word_set = {id(w) for w in hard_s_words}
        orig_hard_s_indices = [
            i for i, w in enumerate(orig_words) if id(w) in hard_s_word_set
        ]

        if len(orig_words) == len(resynth_words):
            # Index matching — simple case
            return [i for i in orig_hard_s_indices if i < len(resynth_words)]

        # Word count mismatch — fall back to text matching
        logger.warning(
            "Word count mismatch: original=%d, resynthesized=%d. Using text matching.",
            len(orig_words), len(resynth_words),
        )
        resynth_word_map: dict[str, list[int]] = {}
        for i, w in enumerate(resynth_words):
            resynth_word_map.setdefault(w.word, []).append(i)

        matched = []
        for idx in orig_hard_s_indices:
            word_text = orig_words[idx].word
            candidates = resynth_word_map.get(word_text, [])
            if candidates:
                matched.append(candidates.pop(0))

        return matched

    def process_batch(
        self,
        dataset_name: str,
        split: str,
        output_dir: str | Path,
        audio_column: str = "audio",
        text_column: str = "text",
        ref_audio_path: str | Path | None = None,
        ref_text: str | None = None,
        save_raw_tts: bool = False,
    ) -> BatchResynthesisReport:
        """Process an entire dataset split through the correction pipeline."""
        from datasets import load_dataset

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading dataset %s split=%s...", dataset_name, split)
        ds = load_dataset(dataset_name, split=split)

        report = BatchResynthesisReport(total_files=len(ds))

        # Resolve reference transcript once for the whole batch.
        resolved_ref_text = ref_text
        if (
            ref_audio_path is not None
            and resolved_ref_text is None
            and self.resynthesis_cfg.provide_ref_text
        ):
            logger.info("Transcribing batch reference audio once: %s", ref_audio_path)
            resolved_ref_text = self._get_asr().transcribe(str(ref_audio_path))
            logger.info("Reference ASR result: '%s'", resolved_ref_text)

        for i, example in enumerate(ds):
            audio_data = example[audio_column]
            transcript = example[text_column]

            # HuggingFace audio column provides path or array
            if isinstance(audio_data, dict):
                audio_path = audio_data.get("path")
                if audio_path is None:
                    # Audio is in-memory — save to temp file
                    import soundfile as sf
                    tmp = output_dir / f"_tmp_input_{i}.wav"
                    sf.write(str(tmp), audio_data["array"], audio_data["sampling_rate"])
                    audio_path = str(tmp)
            else:
                audio_path = str(audio_data)

            output_path = output_dir / f"corrected_{i:05d}.wav"

            try:
                result = self.process_file(
                    audio_path,
                    transcript,
                    output_path,
                    ref_audio_path=ref_audio_path,
                    ref_text=resolved_ref_text,
                    save_raw_tts=save_raw_tts,
                )
            except Exception as e:
                logger.error("Error processing file %d: %s", i, e)
                result = ResynthesisResult(
                    input_path=str(audio_path),
                    output_path=str(output_path),
                    transcript=transcript,
                    status=ResynthesisStatus.error,
                    error_message=str(e),
                )

            report.results.append(result)

            if result.status == ResynthesisStatus.success:
                report.successful += 1
            elif result.status == ResynthesisStatus.no_segments_found:
                report.no_segments += 1
                # Copy original unchanged
                if not output_path.exists():
                    shutil.copy2(audio_path, output_path)
            else:
                report.failed += 1

            if (i + 1) % 10 == 0:
                logger.info("Progress: %d/%d files processed", i + 1, report.total_files)

        logger.info(
            "Batch complete: %d total, %d successful, %d no_segments, %d failed",
            report.total_files, report.successful, report.no_segments, report.failed,
        )
        return report
