# sigmatism_fix — Development Guidelines

Guidance for Claude Code (and humans) working in this repo. Last updated: 2026-05-29.

## What this repo is

Real-time **hard-S (sigmatism) speech correction**. It is the *serving /
optimization* sibling of the research repo `disorder_fix` — the pipeline here was
designed and validated there. The mission is narrow:

> Make the existing OmniVoice + GigaAM-v3 correction pipeline fast enough for
> real-time use (Triton + TensorRT, and other latency tricks) **without changing
> what it outputs.**

The governing principles are in `.specify/memory/constitution.md`. The single most
important one: **Numerical Parity Under Optimization** — never let a speedup
silently change the correction quality. Capture a baseline, gate every optimization
against it.

## Architecture (the import closure that actually runs)

Entry point: `scripts/run_gradio_app.py` → builds one `ResynthesisPipeline`
(`src/pipelines/resynthesis_pipeline.py`) and serves the Gradio UI.

`ResynthesisPipeline.process_file()` is the heart. Per request it does:
1. **ASR** (`src/preprocessing/asr.py`, GigaAM-v3) — only if no transcript given.
2. **Align original** — `gigaam` (RNN-T timesteps, default) or `mfa` (phonemes).
3. **Hard-S filter** (`src/preprocessing/hard_s_filter.py`) — phoneme-based (MFA) or
   espeak text-based (gigaam, via `src/preprocessing/phonemizer.py`).
4. **TTS resynth** (`src/inference/omnivoice_adapter.py`, OmniVoice zero-shot clone).
5. **Align resynth** + match words.
6. **Splice** (`src/inference/word_splice.py`, Hann crossfade) — replace only the
   hard-S words in the original waveform.

The two GPU models — **OmniVoice** (TTS) and **GigaAM-v3** (ASR/aligner) — are the
optimization targets. Everything else is light CPU glue.

### What was intentionally NOT copied from disorder_fix

The U-Net / diffusion audio-inpainting research stack — `models/`, `pipelines/fft.py`,
`pipelines/hifi_gan.py`, `pipelines/vocoder.py`, `pipelines/inference_pipeline.py`,
`inference/infer.py`, and all training / labeling / mining / analysis / W&B code.
The Gradio pipeline does not import any of it. If you find yourself needing it,
you probably want a different repo.

## Commands

```bash
# Unit tests (CPU, fast — GPU models + MFA are mocked in tests/conftest.py)
pytest tests/ -q

# Lint / format
ruff check . && ruff format --check .

# Run the demo (needs a GPU + HF_TOKEN)
HF_TOKEN=... python scripts/run_gradio_app.py --port 7860
```

## Running the Gradio app means run *and* validate

When asked to "run the app", do not stop at the `Launching Gradio demo on …` log
line — that is not validation. Drive a real request end-to-end (e.g. via
`gradio_client` against `http://localhost:7860`, `api_name="/on_correct"`) using the
fixtures under `data/test/khovansky/` (a reference clip + a speech clip), confirm the
pipeline returns `status=success` (or `no_segments_found`) and writes a non-empty
corrected WAV, then report done.

- The `mfa` binary must be on PATH for `aligner: "mfa"` (conda env `sigmatism`). The
  default `aligner: "gigaam"` does not need MFA.
- First run downloads OmniVoice + GigaAM-v3 weights (slow); subsequent runs use the
  HF cache.

## Aligner backends (`resynthesis.aligner` in configs/resynthesis.yaml)

- `gigaam` (default): word boundaries from GigaAM-v3 RNN-T encoder timesteps.
  ~150 ms, **no extra model load** (shares the ASR model). Words only, no phonemes —
  hard-S detection falls back to espeak text-level. Pair with `guard_ms: 80`.
- `mfa`: Montreal Forced Aligner. ~150 s/file but phoneme-level. Pair with
  `guard_ms: 0`. Use only when you need phonemes.

Seam artifacts on `gigaam`? Bump `crossfade_ms` (10 → 20) before touching `guard_ms`.

## Working rules for optimization (from the constitution)

- **Parity first.** Before optimizing a stage, capture reference outputs (ASR text,
  alignment boundaries, corrected waveform) on the `data/test/` fixtures. Gate every
  change against them. A faster pipeline that corrects worse is a regression.
- **Measure, don't guess.** Every optimization needs before/after numbers on the same
  hardware: p50, p95, RTF, GPU memory. Record them under `benchmarks/`.
- **One stage at a time.** Keep ASR / aligner / TTS / splice behind their adapters so
  a single stage can move PyTorch → TensorRT → Triton-remote without touching the
  orchestrator.
- **Config, not constants.** Model names, device, dtype, num_step, batch/concurrency
  are config fields — never hardcode them in source.

## Spec-driven development (speckit)

This repo carries the speckit workflow from `disorder_fix`:
`/speckit.constitution`, `/speckit.specify`, `/speckit.clarify`, `/speckit.plan`,
`/speckit.tasks`, `/speckit.implement`, `/speckit.analyze` (see `.claude/commands/`).
Templates and scripts live in `.specify/`. New features go under `specs/NNN-<slug>/`.
After completing a task in a feature's `tasks.md`, commit before moving on.

## Docs to keep current

- `codemap.md` — living architecture map; update when structure changes.
- `CHANGELOG.md` — Keep a Changelog format; update on significant changes.
- `docs/realtime_optimization.md` — the serving/optimization plan and decisions.
