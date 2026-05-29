"""GradioRunner: owns the pipeline singleton and serializes click handling.

See specs/015-gradio-correction-app/plan.md and research.md.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.apps.gradio_app.persistence import (
    RunPaths,
    rename_raw_tts,
    run_id_new,
    write_envelope,
    write_run_directory,
)
from src.apps.gradio_app.schema import AppConfig, RunResultEnvelope

if TYPE_CHECKING:
    from src.pipelines.resynthesis_pipeline import ResynthesisPipeline

logger = logging.getLogger(__name__)


@contextmanager
def _tee_logs_to_file(log_path: Path):
    """Mirror root-logger records into `log_path` for the duration of the block.

    Console output via the root handler set up in `run_gradio_app.py` is
    preserved; this just adds a parallel FileHandler so every pipeline stage
    (ASR, MFA, hard-S filter, OmniVoice) leaves a per-run on-disk trail.
    """
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    prev_level = root.level
    if prev_level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(prev_level)


class GradioRunner:
    """Single-flight adapter between the Gradio UI and `ResynthesisPipeline`."""

    def __init__(self, app_config: AppConfig, pipeline: ResynthesisPipeline) -> None:
        self.app_config = app_config
        self.pipeline = pipeline
        self._lock = threading.Lock()

    def run(
        self,
        speech_path: str | None,
        reference_path: str | None,
        save_raw_tts_ui: bool = False,
    ) -> tuple[RunResultEnvelope, RunPaths | None]:
        """Execute one correction; return (envelope, run_paths) — paths None on refusal."""
        # --- Single-flight guard (FR-008) ------------------------------------
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            env = self._bare_envelope(
                error="busy: a correction is already running",
            )
            return env, None

        try:
            # --- Input guards (FR-012) --------------------------------------
            if not speech_path:
                return self._bare_envelope(error="missing speech audio"), None
            if not reference_path:
                return self._bare_envelope(error="missing reference audio"), None

            run_id = run_id_new()
            created_at = datetime.now(UTC).isoformat()

            # --- Run directory (FR-013 disk-write-failure edge case) --------
            try:
                paths = write_run_directory(
                    self.app_config.runs_root,
                    run_id,
                    speech_path,
                    reference_path,
                )
            except OSError as exc:
                env = self._bare_envelope(
                    run_id=run_id,
                    created_at=created_at,
                    error=f"disk write failed: {exc}",
                )
                return env, None

            # --- Pipeline invocation ----------------------------------------
            speech_transcript: str | None = None
            reference_transcript: str | None = None
            pipeline_result_dict: dict | None = None
            error: str | None = None
            log_path = paths.run_dir / "pipeline.log"
            with _tee_logs_to_file(log_path):
                logger.info("run %s started (speech=%s, reference=%s)",
                            run_id, paths.input_path, paths.reference_path)
                try:
                    # save_raw_tts=True unconditionally: the UI toggle (save_raw_tts_ui)
                    # only controls the player visibility, never the on-disk artifact.
                    result = self.pipeline.process_file(
                        audio_path=paths.input_path,
                        transcript=None,
                        output_path=paths.corrected_path,
                        ref_audio_path=paths.reference_path,
                        ref_text=None,
                        save_raw_tts=True,
                    )
                    rename_raw_tts(paths.run_dir, paths.corrected_path)
                    speech_transcript = getattr(result, "transcript", None) or None
                    reference_transcript = getattr(result, "ref_transcript", None)
                    pipeline_result_dict = result.to_dict()
                    logger.info("run %s finished", run_id)
                except Exception as exc:  # FR-013 pipeline_crashed
                    logger.exception("Pipeline crashed for run %s", run_id)
                    error = repr(exc)

            envelope = RunResultEnvelope(
                run_id=run_id,
                created_at=created_at,
                speech_transcript=speech_transcript,
                reference_transcript=reference_transcript,
                error=error,
                pipeline_result=pipeline_result_dict,
            )
            try:
                write_envelope(paths.result_json_path, envelope)
            except OSError as exc:
                logger.exception("Failed to write envelope for run %s: %s", run_id, exc)

            return envelope, paths
        finally:
            self._lock.release()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _bare_envelope(
        error: str,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> RunResultEnvelope:
        """Build an envelope for a no-pipeline failure (refused/disk-fail/busy)."""
        return RunResultEnvelope(
            run_id=run_id or run_id_new(),
            created_at=created_at or datetime.now(UTC).isoformat(),
            speech_transcript=None,
            reference_transcript=None,
            error=error,
            pipeline_result=None,
        )
