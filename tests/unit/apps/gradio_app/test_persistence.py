"""Unit tests for src/apps/gradio_app/persistence.py (feature 015)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.apps.gradio_app.persistence import (
    rename_raw_tts,
    run_id_new,
    write_envelope,
    write_run_directory,
)
from src.apps.gradio_app.schema import RunResultEnvelope

SCHEMA_PATH = (
    Path(__file__).resolve().parents[4]
    / "specs/015-gradio-correction-app/contracts/result_json.schema.json"
)


def test_run_id_format():
    pattern = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
    for _ in range(20):
        rid = run_id_new()
        assert pattern.match(rid), rid


def test_run_ids_are_unique():
    ids = {run_id_new() for _ in range(100)}
    assert len(ids) == 100


def test_write_run_directory_creates_layout(tmp_path):
    speech = tmp_path / "speech_src.wav"
    ref = tmp_path / "ref_src.wav"
    speech.write_bytes(b"RIFF....speech")
    ref.write_bytes(b"RIFF....reference")
    runs_root = tmp_path / "app_runs"

    rid = run_id_new()
    paths = write_run_directory(runs_root, rid, speech, ref)

    assert paths.run_dir == runs_root / rid
    assert paths.run_dir.is_dir()
    assert paths.input_path.read_bytes() == b"RIFF....speech"
    assert paths.reference_path.read_bytes() == b"RIFF....reference"
    # Corrected / raw_tts / result.json are paths but not yet materialized.
    assert not paths.corrected_path.exists()
    assert not paths.raw_tts_path.exists()
    assert not paths.result_json_path.exists()


def test_write_run_directory_refuses_existing(tmp_path):
    speech = tmp_path / "s.wav"
    ref = tmp_path / "r.wav"
    speech.write_bytes(b"a")
    ref.write_bytes(b"b")
    runs_root = tmp_path / "app_runs"
    rid = run_id_new()
    write_run_directory(runs_root, rid, speech, ref)
    with pytest.raises(FileExistsError):
        write_run_directory(runs_root, rid, speech, ref)


def test_rename_raw_tts_renames_when_present(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    corrected = run_dir / "corrected.wav"
    corrected.write_bytes(b"final")
    raw = run_dir / "corrected.raw_tts.wav"
    raw.write_bytes(b"raw")

    rename_raw_tts(run_dir, corrected)

    assert not raw.exists()
    assert (run_dir / "raw_tts.wav").read_bytes() == b"raw"


def test_rename_raw_tts_is_noop_when_absent(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    corrected = run_dir / "corrected.wav"
    # no raw_tts file created
    rename_raw_tts(run_dir, corrected)  # should not raise
    assert not (run_dir / "raw_tts.wav").exists()


def _make_envelope(**overrides) -> RunResultEnvelope:
    base = dict(
        run_id=run_id_new(),
        created_at=datetime.now(UTC).isoformat(),
        speech_transcript="привет",
        reference_transcript="эталон",
        error=None,
        pipeline_result={
            "input_path": "/tmp/input.wav",
            "output_path": "/tmp/corrected.wav",
            "status": "success",
            "words_detected": 3,
            "words_corrected": 2,
            "word_details": [
                {
                    "word": "шесть",
                    "original_start": 0.1,
                    "original_end": 0.5,
                    "duration_delta_ms": 12.5,
                }
            ],
            "error_message": None,
        },
    )
    base.update(overrides)
    return RunResultEnvelope(**base)


def test_write_envelope_round_trip(tmp_path):
    out = tmp_path / "result.json"
    env = _make_envelope()
    write_envelope(out, env)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["run_id"] == env.run_id
    assert loaded["pipeline_result"]["status"] == "success"
    assert loaded["error"] is None


def test_envelope_matches_json_schema_success(tmp_path):
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    env = _make_envelope()
    out = tmp_path / "result.json"
    write_envelope(out, env)
    doc = json.loads(out.read_text(encoding="utf-8"))
    errors = sorted(validator.iter_errors(doc), key=lambda e: e.path)
    assert not errors, [e.message for e in errors]


def test_envelope_matches_json_schema_error_path(tmp_path):
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    env = _make_envelope(error="disk write failed", pipeline_result=None)
    out = tmp_path / "result.json"
    write_envelope(out, env)
    doc = json.loads(out.read_text(encoding="utf-8"))
    errors = sorted(validator.iter_errors(doc), key=lambda e: e.path)
    assert not errors, [e.message for e in errors]
