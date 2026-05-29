"""Input-validation tests for GradioRunner (feature 015, FR-012)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.apps.gradio_app.runner import GradioRunner
from src.apps.gradio_app.schema import AppConfig


def _runner(tmp_path: Path) -> GradioRunner:
    cfg = AppConfig(runs_root=tmp_path / "app_runs")
    pipeline = MagicMock(name="ResynthesisPipeline")
    return GradioRunner(cfg, pipeline)


def test_missing_speech_refuses_without_run_dir(tmp_path):
    runner = _runner(tmp_path)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"ref")

    envelope, paths = runner.run(speech_path=None, reference_path=str(ref))

    assert paths is None
    assert envelope.error == "missing speech audio"
    assert envelope.pipeline_result is None
    runner.pipeline.process_file.assert_not_called()
    assert not (tmp_path / "app_runs").exists() or not any((tmp_path / "app_runs").iterdir())


def test_missing_reference_refuses_without_run_dir(tmp_path):
    runner = _runner(tmp_path)
    speech = tmp_path / "speech.wav"
    speech.write_bytes(b"sp")

    envelope, paths = runner.run(speech_path=str(speech), reference_path="")

    assert paths is None
    assert envelope.error == "missing reference audio"
    assert envelope.pipeline_result is None
    runner.pipeline.process_file.assert_not_called()
    assert not (tmp_path / "app_runs").exists() or not any((tmp_path / "app_runs").iterdir())


def test_both_missing_refuses_for_speech_first(tmp_path):
    runner = _runner(tmp_path)
    envelope, paths = runner.run(speech_path=None, reference_path=None)
    assert paths is None
    assert envelope.error == "missing speech audio"
    assert envelope.pipeline_result is None
