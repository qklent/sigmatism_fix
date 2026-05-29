"""Gradio Blocks interface wiring for the hard-S correction demo (feature 015)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from src.apps.gradio_app.runner import GradioRunner

logger = logging.getLogger(__name__)


REFERENCE_GUIDANCE = (
    "Reference should include some ш/ж/щ/ч sounds for best results."
)



def _format_details(envelope) -> str:
    pr = envelope.pipeline_result
    lines: list[str] = []
    lines.append(f"**Speech transcript:** {envelope.speech_transcript or '—'}")
    lines.append(f"**Reference transcript:** {envelope.reference_transcript or '—'}")
    if pr is None:
        if envelope.error:
            lines.append("")
            lines.append(f"_Pipeline did not run: {envelope.error}_")
        return "\n\n".join(lines)
    lines.append(f"**words_detected:** {pr.get('words_detected', 0)}")
    lines.append(f"**words_corrected:** {pr.get('words_corrected', 0)}")
    corrected_words = [w.get("word", "") for w in pr.get("word_details", [])]
    if corrected_words:
        lines.append("**Corrected words:** " + ", ".join(corrected_words))
    else:
        lines.append("**Corrected words:** —")
    if pr.get("error_message"):
        lines.append(f"_Pipeline note: {pr['error_message']}_")
    return "\n\n".join(lines)


def _format_status(envelope) -> str:
    if envelope.error:
        return f"**Error:** {envelope.error}"
    pr = envelope.pipeline_result or {}
    return (
        f"**Done.** status=`{pr.get('status', '?')}`, "
        f"words_detected={pr.get('words_detected', 0)}, "
        f"words_corrected={pr.get('words_corrected', 0)}"
    )


def build_interface(runner: GradioRunner) -> gr.Blocks:
    """Construct the Gradio Blocks demo wired to `runner.run(...)`."""

    def on_correct(reference_path, speech_path, show_raw_tts):
        envelope, paths = runner.run(
            speech_path=speech_path,
            reference_path=reference_path,
            save_raw_tts_ui=bool(show_raw_tts),
        )
        status_md = _format_status(envelope)
        details_md = _format_details(envelope)

        corrected_value = None
        raw_tts_update = gr.update(value=None, visible=False)
        if paths is not None:
            pr = envelope.pipeline_result or {}
            if pr.get("status") in ("success", "no_segments_found") and paths.corrected_path.exists():
                corrected_value = str(paths.corrected_path)
            if paths.raw_tts_path.exists():
                raw_tts_update = gr.update(
                    value=str(paths.raw_tts_path), visible=bool(show_raw_tts)
                )
        return status_md, corrected_value, raw_tts_update, details_md

    with gr.Blocks(title="Hard-S Correction Demo") as demo:
        gr.Markdown("# Hard-S Correction")
        status = gr.Markdown("_Submit a reference clip and a speech clip to fix, then click Correct._")

        with gr.Row():
            with gr.Column():
                reference_audio = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                gr.Markdown(REFERENCE_GUIDANCE)
            with gr.Column():
                speech_audio = gr.Audio(
                    label="Speech to Fix",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

        show_raw_tts = gr.Checkbox(label="Show raw TTS", value=False)
        correct_btn = gr.Button("Correct", variant="primary")

        corrected_audio = gr.Audio(
            label="Corrected Audio",
            type="filepath",
            interactive=False,
        )
        raw_tts_audio = gr.Audio(
            label="Raw TTS (pre-splice)",
            type="filepath",
            interactive=False,
            visible=False,
        )
        details = gr.Markdown("")

        # US3: disable button while in-flight, re-enable after.
        correct_btn.click(
            fn=lambda: gr.update(interactive=False, value="Busy…"),
            inputs=None,
            outputs=correct_btn,
            queue=False,
        ).then(
            fn=on_correct,
            inputs=[reference_audio, speech_audio, show_raw_tts],
            outputs=[status, corrected_audio, raw_tts_audio, details],
        ).then(
            fn=lambda: gr.update(interactive=True, value="Correct"),
            inputs=None,
            outputs=correct_btn,
            queue=False,
        )

    # Belt-and-braces: Gradio's own serialization in addition to the runner lock.
    demo.queue(default_concurrency_limit=1)
    return demo
