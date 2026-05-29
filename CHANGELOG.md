# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); this project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository bootstrap. Copied the validated **resynthesis (hard-S
  correction) pipeline** from `disorder_fix` — the exact import closure used by the
  Gradio demo:
  - Orchestration: `src/pipelines/resynthesis_pipeline.py`.
  - Stages: GigaAM-v3 ASR (`src/preprocessing/asr.py`), GigaAM RNN-T + MFA aligners,
    hard-S filter + espeak phonemizer, OmniVoice TTS adapter, word-level splicer.
  - Gradio app (`src/apps/gradio_app/`) + launcher (`scripts/run_gradio_app.py`).
  - Config schema + `configs/resynthesis.yaml`, `configs/default.yaml`.
  - CPU unit-test suite mirroring `src/` (GPU models + MFA mocked) — the correctness
    baseline for parity-gated optimization.
  - Audio fixtures under `data/test/` for smoke / latency benchmarking.
- Project docs: `README.md`, `CLAUDE.md`, `codemap.md`,
  `docs/realtime_optimization.md`.
- Spec-driven development tooling (speckit): `.specify/` templates + scripts and
  `.claude/commands/speckit.*`. New project constitution at
  `.specify/memory/constitution.md` (v1.0.0) — adapted from disorder_fix and
  reframed for real-time serving (parity-under-optimization, latency budgets, GPU
  serving discipline).

### Excluded (intentionally not copied from disorder_fix)
- Training / labeling / mining / analysis / evaluation / W&B code.
- The U-Net / diffusion / HiFi-GAN audio-inpainting stack — not used by the Gradio
  resynthesis pipeline.

### Next
- Profile the PyTorch baseline; establish per-stage latency budgets.
- Build a numerical-parity harness over `data/test/` fixtures.
- Export GigaAM-v3 + OmniVoice to ONNX → TensorRT; serve via Triton.
