"""Unit test: ref_audio override forwards to the TTS adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from src.config.schema import OmniVoiceConfig, ResynthesisConfig
from src.pipelines.resynthesis_pipeline import ResynthesisPipeline


def _mock_word(word: str, start: float, end: float, phonemes=None):
    w = MagicMock()
    w.word = word
    w.start = start
    w.end = end
    w.phonemes = phonemes or []
    return w


def _build_pipeline() -> ResynthesisPipeline:
    with (
        patch("src.pipelines.resynthesis_pipeline.MFAAligner") as mock_aligner,
        patch("src.pipelines.resynthesis_pipeline.OmniVoiceAdapter") as mock_tts,
    ):
        mock_aligner.return_value = MagicMock()
        mock_tts.return_value = MagicMock()
        pipeline = ResynthesisPipeline(
            OmniVoiceConfig(),
            ResynthesisConfig(provide_ref_text=True),
            device="cpu",
        )
    return pipeline


def test_process_file_forwards_ref_audio_to_adapter(tmp_path: Path):
    pipeline = _build_pipeline()

    audio_path = tmp_path / "input.wav"
    ref_path = tmp_path / "ref.wav"
    output_path = tmp_path / "out.wav"
    audio_path.write_bytes(b"")
    ref_path.write_bytes(b"")

    orig_words = [_mock_word("саша", 0.0, 0.5)]
    resynth_words = [_mock_word("саша", 0.0, 0.5)]

    pipeline.aligner.align_from_file_full = MagicMock(
        return_value=([], orig_words)
    )

    resynth_wave = torch.zeros(1, 8000)
    pipeline.tts.synthesize = MagicMock(return_value=(resynth_wave, 16000))

    with (
        patch(
            "src.pipelines.resynthesis_pipeline.torchaudio.load",
            return_value=(torch.zeros(1, 16000), 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.filter_hard_s_words",
            return_value=orig_words,
        ),
        patch("src.pipelines.resynthesis_pipeline.save_waveform"),
        patch(
            "src.pipelines.resynthesis_pipeline.splice_words",
            return_value=torch.zeros(1, 16000),
        ),
    ):
        # Second align call (on the resynthesized temp wav) returns resynth_words.
        pipeline.aligner.align_from_file_full.side_effect = [
            ([], orig_words),
            ([], resynth_words),
        ]
        pipeline.process_file(
            audio_path,
            transcript="саша",
            output_path=output_path,
            ref_audio_path=ref_path,
            ref_text="clean reference text",
        )

    pipeline.tts.synthesize.assert_called_once()
    call_kwargs = pipeline.tts.synthesize.call_args.kwargs
    assert call_kwargs["ref_audio_path"] == str(ref_path)
    assert call_kwargs["ref_text"] == "clean reference text"


def test_process_file_self_reference_when_ref_audio_none(tmp_path: Path):
    pipeline = _build_pipeline()

    audio_path = tmp_path / "input.wav"
    output_path = tmp_path / "out.wav"
    audio_path.write_bytes(b"")

    orig_words = [_mock_word("саша", 0.0, 0.5)]
    resynth_words = [_mock_word("саша", 0.0, 0.5)]

    pipeline.aligner.align_from_file_full = MagicMock(
        side_effect=[([], orig_words), ([], resynth_words)]
    )
    pipeline.tts.synthesize = MagicMock(
        return_value=(torch.zeros(1, 8000), 16000)
    )

    with (
        patch(
            "src.pipelines.resynthesis_pipeline.torchaudio.load",
            return_value=(torch.zeros(1, 16000), 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.filter_hard_s_words",
            return_value=orig_words,
        ),
        patch("src.pipelines.resynthesis_pipeline.save_waveform"),
        patch(
            "src.pipelines.resynthesis_pipeline.splice_words",
            return_value=torch.zeros(1, 16000),
        ),
    ):
        pipeline.process_file(
            audio_path,
            transcript="саша",
            output_path=output_path,
        )

    call_kwargs = pipeline.tts.synthesize.call_args.kwargs
    assert call_kwargs["ref_audio_path"] == str(audio_path)
    assert call_kwargs["ref_text"] == "саша"
