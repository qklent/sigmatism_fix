"""Unit tests for ResynthesisPipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from src.config.schema import OmniVoiceConfig, ResynthesisConfig
from src.pipelines.resynthesis_pipeline import (
    BatchResynthesisReport,
    CorrectedWord,
    ResynthesisPipeline,
    ResynthesisResult,
    ResynthesisStatus,
    _NullProbe,
)


# -- Dataclasses & enums ------------------------------------------------------


def test_resynthesis_status_enum_values():
    assert ResynthesisStatus.success.value == "success"
    assert ResynthesisStatus.no_segments_found.value == "no_segments_found"
    assert ResynthesisStatus.alignment_failed.value == "alignment_failed"


def test_resynthesis_result_to_dict_serializes_word_details():
    result = ResynthesisResult(
        input_path="in.wav",
        output_path="out.wav",
        transcript="тест",
        ref_transcript="ref",
        words_detected=1,
        words_corrected=1,
        word_details=[
            CorrectedWord(
                word="тест",
                original_start=0.0,
                original_end=0.5,
                resynth_start=0.0,
                resynth_end=0.5,
                duration_delta_ms=0.0,
            )
        ],
        status=ResynthesisStatus.success,
        error_message=None,
    )
    d = result.to_dict()
    assert d["status"] == "success"
    assert d["ref_transcript"] == "ref"
    assert d["words_corrected"] == 1
    assert d["word_details"][0]["word"] == "тест"
    assert "resynth_start" not in d["word_details"][0]  # only the documented keys


def test_batch_report_to_dict_failure_rate():
    report = BatchResynthesisReport(total_files=10, successful=7, no_segments=2, failed=1)
    d = report.to_dict()
    assert d["failure_rate"] == 0.1


def test_batch_report_to_dict_zero_files_no_div_by_zero():
    report = BatchResynthesisReport(total_files=0)
    assert report.to_dict()["failure_rate"] == 0.0


def test_null_probe_returns_nullcontexts():
    probe = _NullProbe()
    with probe.stage("alpha", gpu=True):
        pass
    with probe.end_to_end():
        pass


# -- Pipeline construction ----------------------------------------------------


def _build_pipeline(aligner="mfa", provide_ref_text=True):
    with (
        patch("src.pipelines.resynthesis_pipeline.MFAAligner") as mock_aligner,
        patch("src.pipelines.resynthesis_pipeline.OmniVoiceAdapter") as mock_tts,
    ):
        mock_aligner.return_value = MagicMock()
        mock_tts.return_value = MagicMock()
        return ResynthesisPipeline(
            OmniVoiceConfig(),
            ResynthesisConfig(aligner=aligner, provide_ref_text=provide_ref_text),
            device="cpu",
        )


def test_init_mfa_branch_constructs_real_aligner():
    pipe = _build_pipeline(aligner="mfa")
    assert pipe._aligner_kind == "mfa"
    assert pipe.aligner is not None


def test_init_gigaam_branch_leaves_aligner_lazy():
    pipe = _build_pipeline(aligner="gigaam")
    assert pipe._aligner_kind == "gigaam"
    assert pipe.aligner is None


def test_init_device_falls_back_when_none(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with (
        patch("src.pipelines.resynthesis_pipeline.MFAAligner"),
        patch("src.pipelines.resynthesis_pipeline.OmniVoiceAdapter"),
    ):
        pipe = ResynthesisPipeline(OmniVoiceConfig(), ResynthesisConfig(), device=None)
    assert pipe.device == "cpu"


def test_get_asr_is_memoized():
    pipe = _build_pipeline(aligner="mfa")
    with patch("src.preprocessing.asr.GigaAMTranscriber") as mock_asr_cls:
        mock_asr_cls.return_value = MagicMock()
        first = pipe._get_asr()
        second = pipe._get_asr()
    assert first is second
    assert mock_asr_cls.call_count == 1


def test_get_gigaam_aligner_builds_once():
    pipe = _build_pipeline(aligner="gigaam")
    pipe._asr = MagicMock()
    with patch("src.preprocessing.gigaam_aligner.GigaAMAligner") as mock_cls:
        mock_cls.return_value = MagicMock()
        first = pipe._get_gigaam_aligner()
        second = pipe._get_gigaam_aligner()
    assert first is second
    assert mock_cls.call_count == 1


# -- _match_target_indices ----------------------------------------------------


def _w(word: str):
    m = MagicMock()
    m.word = word
    return m


def test_match_target_indices_same_length_uses_index_match():
    pipe = _build_pipeline()
    orig = [_w("a"), _w("b"), _w("c")]
    resynth = [_w("a"), _w("b"), _w("c")]
    hard_s = [orig[1]]
    out = pipe._match_target_indices(orig, resynth, hard_s)
    assert out == [1]


def test_match_target_indices_length_mismatch_falls_back_to_text():
    pipe = _build_pipeline()
    orig = [_w("саша"), _w("шла"), _w("по")]
    resynth = [_w("саша"), _w("по"), _w("шла"), _w("шла")]
    hard_s = [orig[0], orig[1]]
    out = pipe._match_target_indices(orig, resynth, hard_s)
    # "саша" → idx 0 in resynth, "шла" → first idx 2 in resynth
    assert out == [0, 2]


def test_match_target_indices_text_fallback_skips_unmatched():
    """Length mismatch + no text overlap → empty matched list."""
    pipe = _build_pipeline()
    orig = [_w("саша"), _w("шла")]
    resynth = [_w("only"), _w("here"), _w("nothing")]  # different length, no overlap
    hard_s = [orig[0]]
    out = pipe._match_target_indices(orig, resynth, hard_s)
    assert out == []


# -- process_file error paths -------------------------------------------------


def test_process_file_asr_failure_returns_error_status(tmp_path: Path):
    pipe = _build_pipeline()
    with patch.object(pipe, "_get_asr") as mock_get_asr:
        mock_get_asr.return_value.transcribe.side_effect = RuntimeError("boom")
        result = pipe.process_file(
            tmp_path / "in.wav", transcript=None, output_path=tmp_path / "out.wav"
        )
    assert result.status == ResynthesisStatus.error
    assert "ASR" in result.error_message


def test_process_file_audio_load_failure_returns_error(tmp_path: Path):
    pipe = _build_pipeline()
    audio_path = tmp_path / "in.wav"
    audio_path.write_bytes(b"not-a-wav")
    with patch(
        "src.pipelines.resynthesis_pipeline.torchaudio.load",
        side_effect=RuntimeError("cannot decode"),
    ):
        result = pipe.process_file(
            audio_path, transcript="тест", output_path=tmp_path / "out.wav"
        )
    assert result.status == ResynthesisStatus.error
    assert "load audio" in result.error_message.lower()


def test_process_file_alignment_failure(tmp_path: Path):
    pipe = _build_pipeline()
    audio_path = tmp_path / "in.wav"
    audio_path.write_bytes(b"")
    pipe.aligner.align_from_file_full = MagicMock(side_effect=RuntimeError("MFA crashed"))
    with patch(
        "src.pipelines.resynthesis_pipeline.torchaudio.load",
        return_value=(torch.zeros(1, 16000), 16000),
    ):
        result = pipe.process_file(
            audio_path, transcript="тест", output_path=tmp_path / "out.wav"
        )
    assert result.status == ResynthesisStatus.alignment_failed


def test_process_file_no_hard_s_copies_original(tmp_path: Path):
    pipe = _build_pipeline()
    audio_path = tmp_path / "in.wav"
    audio_path.write_bytes(b"original-bytes")
    output_path = tmp_path / "out.wav"
    pipe.aligner.align_from_file_full = MagicMock(return_value=([], []))
    with (
        patch(
            "src.pipelines.resynthesis_pipeline.torchaudio.load",
            return_value=(torch.zeros(1, 16000), 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.filter_hard_s_words",
            return_value=[],
        ),
    ):
        result = pipe.process_file(audio_path, transcript="тест", output_path=output_path)
    assert result.status == ResynthesisStatus.no_segments_found
    assert output_path.exists()
    assert output_path.read_bytes() == b"original-bytes"


def test_process_file_resynthesis_failure(tmp_path: Path):
    pipe = _build_pipeline()
    audio_path = tmp_path / "in.wav"
    audio_path.write_bytes(b"")
    orig = [_w("саша")]
    pipe.aligner.align_from_file_full = MagicMock(return_value=([], orig))
    pipe.tts.synthesize = MagicMock(side_effect=RuntimeError("TTS exploded"))
    with (
        patch(
            "src.pipelines.resynthesis_pipeline.torchaudio.load",
            return_value=(torch.zeros(1, 16000), 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.filter_hard_s_words",
            return_value=orig,
        ),
    ):
        result = pipe.process_file(
            audio_path, transcript="саша", output_path=tmp_path / "out.wav"
        )
    assert result.status == ResynthesisStatus.resynthesis_failed
    assert "Resynthesis failed" in result.error_message


def test_process_file_save_raw_tts_writes_extra_file(tmp_path: Path):
    pipe = _build_pipeline()
    audio_path = tmp_path / "in.wav"
    audio_path.write_bytes(b"")
    output_path = tmp_path / "corrected.wav"

    orig = [_w("саша")]
    pipe.aligner.align_from_file_full = MagicMock(
        side_effect=[([], orig), ([], orig)]
    )
    pipe.tts.synthesize = MagicMock(return_value=(torch.zeros(1, 16000), 16000))

    saved_paths = []

    def _spy_save(wave, path, sr):
        saved_paths.append(Path(path))
        Path(path).write_bytes(b"")

    with (
        patch(
            "src.pipelines.resynthesis_pipeline.torchaudio.load",
            return_value=(torch.zeros(1, 16000), 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.filter_hard_s_words",
            return_value=orig,
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.splice_words",
            return_value=torch.zeros(1, 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.save_waveform",
            side_effect=_spy_save,
        ),
    ):
        result = pipe.process_file(
            audio_path,
            transcript="саша",
            output_path=output_path,
            save_raw_tts=True,
        )
    assert result.status == ResynthesisStatus.success
    raw_tts_paths = [p for p in saved_paths if p.name.endswith(".raw_tts.wav")]
    assert raw_tts_paths, f"expected a .raw_tts.wav save; got {saved_paths}"


def test_process_file_resamples_to_configured_output_sr(tmp_path: Path):
    pipe = _build_pipeline()
    pipe.resynthesis_cfg.output_sample_rate = 22050
    audio_path = tmp_path / "in.wav"
    audio_path.write_bytes(b"")

    orig = [_w("a")]
    pipe.aligner.align_from_file_full = MagicMock(
        side_effect=[([], orig), ([], orig)]
    )
    pipe.tts.synthesize = MagicMock(return_value=(torch.zeros(1, 16000), 16000))

    saved_srs: list[int] = []

    def _spy_save(wave, path, sr):
        saved_srs.append(sr)

    with (
        patch(
            "src.pipelines.resynthesis_pipeline.torchaudio.load",
            return_value=(torch.zeros(1, 16000), 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.filter_hard_s_words",
            return_value=orig,
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.splice_words",
            return_value=torch.zeros(1, 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.save_waveform",
            side_effect=_spy_save,
        ),
    ):
        pipe.process_file(audio_path, transcript="a", output_path=tmp_path / "out.wav")

    # The final save should be at 22050 Hz.
    assert 22050 in saved_srs


def test_process_file_match_failure_copies_original(tmp_path: Path):
    """When _match_target_indices returns [], the pipeline copies the original.

    Triggered by a length mismatch between orig and resynth alignment plus no
    text overlap with the hard-S word.
    """
    pipe = _build_pipeline()
    audio_path = tmp_path / "in.wav"
    audio_path.write_bytes(b"original")
    output_path = tmp_path / "out.wav"

    orig = [_w("саша")]
    resynth_words = [_w("totally"), _w("different"), _w("words")]  # different length
    pipe.aligner.align_from_file_full = MagicMock(
        side_effect=[([], orig), ([], resynth_words)]
    )
    pipe.tts.synthesize = MagicMock(return_value=(torch.zeros(1, 16000), 16000))

    with (
        patch(
            "src.pipelines.resynthesis_pipeline.torchaudio.load",
            return_value=(torch.zeros(1, 16000), 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.filter_hard_s_words",
            return_value=orig,
        ),
        patch("src.pipelines.resynthesis_pipeline.save_waveform"),
    ):
        result = pipe.process_file(
            audio_path, transcript="саша", output_path=output_path
        )

    assert result.status == ResynthesisStatus.no_segments_found
    assert output_path.exists()


def test_process_file_no_ref_text_with_provide_ref_text_calls_asr_on_reference(tmp_path: Path):
    pipe = _build_pipeline(provide_ref_text=True)
    audio_path = tmp_path / "in.wav"
    ref_path = tmp_path / "ref.wav"
    output_path = tmp_path / "out.wav"
    audio_path.write_bytes(b"")
    ref_path.write_bytes(b"")

    orig = [_w("саша")]
    pipe.aligner.align_from_file_full = MagicMock(
        side_effect=[([], orig), ([], orig)]
    )
    pipe.tts.synthesize = MagicMock(return_value=(torch.zeros(1, 16000), 16000))

    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = "transcribed-reference"
    with (
        patch.object(pipe, "_get_asr", return_value=fake_asr),
        patch(
            "src.pipelines.resynthesis_pipeline.torchaudio.load",
            return_value=(torch.zeros(1, 16000), 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.filter_hard_s_words",
            return_value=orig,
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.splice_words",
            return_value=torch.zeros(1, 16000),
        ),
        patch("src.pipelines.resynthesis_pipeline.save_waveform"),
    ):
        result = pipe.process_file(
            audio_path,
            transcript="саша",
            output_path=output_path,
            ref_audio_path=ref_path,
            ref_text=None,
        )
    assert result.ref_transcript == "transcribed-reference"
    fake_asr.transcribe.assert_called_once_with(str(ref_path))


def test_process_batch_aggregates_results(tmp_path: Path):
    """process_batch walks a fake dataset and produces a BatchResynthesisReport."""
    pipe = _build_pipeline()

    # Each example just routes through process_file, which we monkey-patch.
    def _fake_process_file(audio_path, transcript, output_path, **kwargs):
        Path(output_path).write_bytes(b"out")
        return ResynthesisResult(
            input_path=str(audio_path),
            output_path=str(output_path),
            transcript=transcript,
            status=ResynthesisStatus.success,
        )

    pipe.process_file = _fake_process_file

    fake_dataset = [
        {"audio": str(tmp_path / "a.wav"), "text": "one"},
        {"audio": str(tmp_path / "b.wav"), "text": "two"},
    ]
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.wav").write_bytes(b"")

    with patch("datasets.load_dataset", return_value=fake_dataset):
        report = pipe.process_batch(
            dataset_name="ignored",
            split="train",
            output_dir=tmp_path / "outputs",
        )
    assert report.total_files == 2
    assert report.successful == 2
    assert report.failed == 0


def test_process_batch_handles_dict_audio_with_in_memory_array(tmp_path: Path):
    """When audio is a dict with no path, the batch falls back to a tmp WAV."""
    pipe = _build_pipeline()

    written: list[Path] = []

    def _fake_process_file(audio_path, transcript, output_path, **kwargs):
        written.append(Path(audio_path))
        Path(output_path).write_bytes(b"out")
        return ResynthesisResult(
            input_path=str(audio_path),
            output_path=str(output_path),
            transcript=transcript,
            status=ResynthesisStatus.no_segments_found,
        )

    pipe.process_file = _fake_process_file

    import numpy as np

    fake_dataset = [
        {
            "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
            "text": "one",
        }
    ]
    with (
        patch("datasets.load_dataset", return_value=fake_dataset),
        patch("soundfile.write") as mock_write,
    ):
        report = pipe.process_batch(
            dataset_name="ignored",
            split="train",
            output_dir=tmp_path / "outputs",
        )
    assert mock_write.called
    assert report.no_segments == 1


def test_process_batch_records_failure_for_raising_process_file(tmp_path: Path):
    """A process_file exception is caught, recorded as failed in the report."""
    pipe = _build_pipeline()
    pipe.process_file = MagicMock(side_effect=RuntimeError("kapow"))

    fake_dataset = [{"audio": str(tmp_path / "a.wav"), "text": "one"}]
    (tmp_path / "a.wav").write_bytes(b"")
    with patch("datasets.load_dataset", return_value=fake_dataset):
        report = pipe.process_batch(
            dataset_name="ignored",
            split="train",
            output_dir=tmp_path / "outputs",
        )
    assert report.failed == 1
    assert report.results[0].status == ResynthesisStatus.error


def test_process_batch_resolves_ref_text_via_asr_once(tmp_path: Path):
    """A batch with ref_audio_path + provide_ref_text=True transcribes once."""
    pipe = _build_pipeline(provide_ref_text=True)
    pipe.process_file = MagicMock(
        return_value=ResynthesisResult(
            input_path="x", output_path="y", status=ResynthesisStatus.success
        )
    )
    ref_path = tmp_path / "ref.wav"
    ref_path.write_bytes(b"")
    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = "resolved-once"

    fake_dataset = [
        {"audio": str(tmp_path / "a.wav"), "text": "one"},
        {"audio": str(tmp_path / "b.wav"), "text": "two"},
    ]
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.wav").write_bytes(b"")

    with (
        patch.object(pipe, "_get_asr", return_value=fake_asr),
        patch("datasets.load_dataset", return_value=fake_dataset),
    ):
        pipe.process_batch(
            dataset_name="ignored",
            split="train",
            output_dir=tmp_path / "outputs",
            ref_audio_path=ref_path,
        )
    # Reference transcription happens exactly once for the whole batch.
    fake_asr.transcribe.assert_called_once_with(str(ref_path))


def test_align_dispatch_routes_to_gigaam_branch():
    """_align uses GigaAMAligner when configured."""
    pipe = _build_pipeline(aligner="gigaam")
    mock_aligner = MagicMock()
    mock_aligner.align_from_file_full.return_value = ([], [])
    with patch.object(pipe, "_get_gigaam_aligner", return_value=mock_aligner):
        pipe._align("/x.wav", "test")
    mock_aligner.align_from_file_full.assert_called_once_with("/x.wav", "test")


def test_process_file_provide_ref_text_false_passes_none(tmp_path: Path):
    pipe = _build_pipeline(provide_ref_text=False)
    audio_path = tmp_path / "in.wav"
    ref_path = tmp_path / "ref.wav"
    audio_path.write_bytes(b"")
    ref_path.write_bytes(b"")

    orig = [_w("саша")]
    pipe.aligner.align_from_file_full = MagicMock(
        side_effect=[([], orig), ([], orig)]
    )
    pipe.tts.synthesize = MagicMock(return_value=(torch.zeros(1, 16000), 16000))

    with (
        patch(
            "src.pipelines.resynthesis_pipeline.torchaudio.load",
            return_value=(torch.zeros(1, 16000), 16000),
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.filter_hard_s_words",
            return_value=orig,
        ),
        patch(
            "src.pipelines.resynthesis_pipeline.splice_words",
            return_value=torch.zeros(1, 16000),
        ),
        patch("src.pipelines.resynthesis_pipeline.save_waveform"),
    ):
        pipe.process_file(
            audio_path,
            transcript="саша",
            output_path=tmp_path / "out.wav",
            ref_audio_path=ref_path,
            ref_text="ignored-because-flag-is-false",
        )

    assert pipe.tts.synthesize.call_args.kwargs["ref_text"] is None
