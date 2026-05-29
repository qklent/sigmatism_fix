"""On-disk persistence for Gradio correction-app runs (feature 015).

Layout contract: specs/015-gradio-correction-app/contracts/run_directory.md
Envelope contract: specs/015-gradio-correction-app/contracts/result_json.schema.json
"""

from __future__ import annotations

import os
import secrets
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.apps.gradio_app.schema import RunResultEnvelope


@dataclass(frozen=True)
class RunPaths:
    """Canonical absolute paths for a single run directory."""

    run_dir: Path
    input_path: Path
    reference_path: Path
    corrected_path: Path
    raw_tts_path: Path
    result_json_path: Path


def run_id_new() -> str:
    """Return `YYYYmmdd-HHMMSS-xxxxxx` (UTC + 6 hex chars)."""
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(3)}"


def write_run_directory(
    runs_root: Path,
    run_id: str,
    speech_src_path: str | Path,
    reference_src_path: str | Path,
) -> RunPaths:
    """Create `runs_root/run_id/` and copy the two user uploads into it.

    Raises OSError on any filesystem failure (callers catch and surface to UI).
    """
    runs_root = Path(runs_root)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    paths = RunPaths(
        run_dir=run_dir,
        input_path=run_dir / "input.wav",
        reference_path=run_dir / "reference.wav",
        corrected_path=run_dir / "corrected.wav",
        raw_tts_path=run_dir / "raw_tts.wav",
        result_json_path=run_dir / "result.json",
    )

    shutil.copyfile(str(speech_src_path), paths.input_path)
    shutil.copyfile(str(reference_src_path), paths.reference_path)

    return paths


def rename_raw_tts(run_dir: Path, corrected_path: Path) -> None:
    """Rename pipeline's `<stem>.raw_tts.wav` to canonical `raw_tts.wav`.

    No-op if the source file is absent (pipeline failed before TTS ran).
    """
    src = corrected_path.with_name(corrected_path.stem + ".raw_tts.wav")
    if src.exists():
        dst = run_dir / "raw_tts.wav"
        os.replace(src, dst)


def write_envelope(result_json_path: Path, envelope: RunResultEnvelope) -> None:
    """Atomic write: envelope → temp file → os.replace."""
    result_json_path = Path(result_json_path)
    tmp = result_json_path.with_suffix(result_json_path.suffix + ".tmp")
    tmp.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, result_json_path)
