"""Single-flight lock test for GradioRunner (feature 015, FR-008, SC-006)."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

from src.apps.gradio_app.runner import GradioRunner
from src.apps.gradio_app.schema import AppConfig


class _BlockingPipeline:
    """Mimics `ResynthesisPipeline` for the lock test.

    `process_file` blocks on a Barrier so the first call holds the runner's
    lock while the main thread invokes `.run(...)` a second time.
    """

    def __init__(self, barrier: threading.Barrier) -> None:
        self.barrier = barrier
        self.release = threading.Event()
        self.calls = 0

    def process_file(self, **kwargs):  # noqa: ARG002
        self.calls += 1
        self.barrier.wait(timeout=5)
        self.release.wait(timeout=5)
        result = MagicMock()
        result.transcript = "x"
        result.ref_transcript = None
        result.to_dict.return_value = {
            "input_path": str(kwargs["audio_path"]),
            "output_path": str(kwargs["output_path"]),
            "status": "success",
            "words_detected": 0,
            "words_corrected": 0,
            "word_details": [],
            "error_message": None,
        }
        # Touch the output path so downstream rename/envelope don't get confused.
        Path(kwargs["output_path"]).write_bytes(b"corrected")
        return result


def test_second_concurrent_run_is_refused(tmp_path):
    barrier = threading.Barrier(2)
    pipeline = _BlockingPipeline(barrier)

    speech = tmp_path / "speech.wav"
    reference = tmp_path / "reference.wav"
    speech.write_bytes(b"sp")
    reference.write_bytes(b"rf")

    cfg = AppConfig(runs_root=tmp_path / "app_runs")
    runner = GradioRunner(cfg, pipeline)

    first_result: dict = {}

    def first_call():
        env, paths = runner.run(speech_path=str(speech), reference_path=str(reference))
        first_result["env"] = env
        first_result["paths"] = paths

    t = threading.Thread(target=first_call)
    t.start()
    # Wait until the mock pipeline has entered process_file and is holding the
    # runner's lock.
    barrier.wait(timeout=5)

    # Second call while the first is in-flight.
    env2, paths2 = runner.run(speech_path=str(speech), reference_path=str(reference))
    assert paths2 is None
    assert env2.error is not None and "busy" in env2.error.lower()

    # Release the first call and let it finish cleanly.
    pipeline.release.set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert pipeline.calls == 1
    assert first_result["paths"] is not None

    # Only one run directory was created.
    run_dirs = list((tmp_path / "app_runs").iterdir())
    assert len(run_dirs) == 1
