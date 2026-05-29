"""Unit tests for the Gradio app's formatters and Blocks builder."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import gradio as gr

from src.apps.gradio_app.app import (
    REFERENCE_GUIDANCE,
    _format_details,
    _format_status,
    build_interface,
)


def _envelope(pipeline_result=None, error=None, speech_transcript="", reference_transcript=""):
    return SimpleNamespace(
        pipeline_result=pipeline_result,
        error=error,
        speech_transcript=speech_transcript,
        reference_transcript=reference_transcript,
    )


def test_format_status_success():
    envelope = _envelope(
        pipeline_result={"status": "success", "words_detected": 3, "words_corrected": 2}
    )
    status = _format_status(envelope)
    assert "status=`success`" in status
    assert "words_detected=3" in status
    assert "words_corrected=2" in status


def test_format_status_error():
    envelope = _envelope(error="something broke")
    assert "Error" in _format_status(envelope)
    assert "something broke" in _format_status(envelope)


def test_format_status_missing_pipeline_result_falls_back():
    """No error and no pipeline_result -> formats with defaults."""
    envelope = _envelope()
    out = _format_status(envelope)
    assert "status=`?`" in out
    assert "words_detected=0" in out


def test_format_details_pipeline_did_not_run():
    envelope = _envelope(error="bad input")
    out = _format_details(envelope)
    assert "Pipeline did not run" in out
    assert "bad input" in out


def test_format_details_no_pipeline_result_no_error_short_circuits():
    envelope = _envelope()  # pipeline_result=None, error=None
    out = _format_details(envelope)
    assert "**Speech transcript:**" in out
    # No 'Pipeline did not run' line — the function returns before that block
    assert "Pipeline did not run" not in out


def test_format_details_with_corrected_words():
    envelope = _envelope(
        pipeline_result={
            "status": "success",
            "words_detected": 2,
            "words_corrected": 2,
            "word_details": [{"word": "саша"}, {"word": "шла"}],
        },
        speech_transcript="Саша шла",
        reference_transcript="Reference",
    )
    out = _format_details(envelope)
    assert "Саша шла" in out
    assert "Reference" in out
    assert "words_detected:** 2" in out
    assert "саша, шла" in out


def test_format_details_no_corrected_words_uses_em_dash():
    envelope = _envelope(
        pipeline_result={
            "status": "no_segments_found",
            "words_detected": 0,
            "words_corrected": 0,
            "word_details": [],
        }
    )
    out = _format_details(envelope)
    assert "Corrected words:** —" in out


def test_format_details_pipeline_error_message_appended():
    envelope = _envelope(
        pipeline_result={
            "status": "alignment_failed",
            "words_detected": 0,
            "words_corrected": 0,
            "word_details": [],
            "error_message": "MFA failed",
        }
    )
    out = _format_details(envelope)
    assert "MFA failed" in out


def test_reference_guidance_constant_mentions_sibilants():
    assert "ш" in REFERENCE_GUIDANCE
    assert "ч" in REFERENCE_GUIDANCE


def test_build_interface_returns_blocks():
    runner = MagicMock()
    demo = build_interface(runner)
    assert isinstance(demo, gr.Blocks)
    # Title is set on the Blocks instance.
    assert "Hard-S" in demo.title
