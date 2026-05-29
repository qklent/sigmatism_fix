# Codemap

Living architecture map. Update when structure changes (constitution, Principle IX).

## Module graph (runtime import closure)

```
scripts/run_gradio_app.py
  └─ src/apps/gradio_app/app.py          build_interface() — Gradio Blocks UI
       └─ src/apps/gradio_app/runner.py  GradioRunner — single-flight, per-run artifacts
            ├─ src/apps/gradio_app/persistence.py   RunPaths, write_envelope, …
            ├─ src/apps/gradio_app/schema.py        AppConfig, RunResultEnvelope (pydantic)
            └─ src/pipelines/resynthesis_pipeline.py   ResynthesisPipeline  ◄── CORE
                 ├─ src/config/schema.py             OmniVoiceConfig, ResynthesisConfig, …
                 ├─ src/data/audio_adapter.py        load/save waveform, mel adapter
                 ├─ src/inference/omnivoice_adapter.py   OmniVoiceAdapter (TTS)  [GPU]
                 ├─ src/inference/word_splice.py      splice_words() — Hann crossfade
                 ├─ src/preprocessing/forced_aligner.py   MFAAligner, WordAlignment, PhonemeAlignment
                 ├─ src/preprocessing/hard_s_filter.py    filter_hard_s_words[_by_text]
                 │     └─ src/preprocessing/phonemizer.py  RussianPhonemizer (espeak-ng)
                 ├─ (lazy) src/preprocessing/asr.py        GigaAMTranscriber (ASR)  [GPU]
                 └─ (lazy) src/preprocessing/gigaam_aligner.py  GigaAMAligner (RNN-T timesteps)
```

`src/utils/{config,seed}.py` and `src/data/masking.py` are general helpers carried
along; not on the hot path of the Gradio request.

## Stages and where the time goes

| Stage | Module | Device | Notes |
|-------|--------|--------|-------|
| ASR | `preprocessing/asr.py` | GPU | GigaAM-v3 RNN-T; skipped if transcript provided |
| Align | `preprocessing/{gigaam_aligner,forced_aligner}.py` | GPU / CPU+subproc | gigaam ~150 ms (reuses ASR) vs mfa ~150 s |
| Hard-S filter | `preprocessing/hard_s_filter.py` | CPU | phoneme (mfa) or espeak text (gigaam) |
| TTS | `inference/omnivoice_adapter.py` | GPU | OmniVoice diffusion; `num_step` is the big knob |
| Splice | `inference/word_splice.py` | CPU | replaces only hard-S words |

**Optimization targets:** the two GPU models (OmniVoice, GigaAM-v3). See
`docs/realtime_optimization.md`.

## Config

- `configs/resynthesis.yaml` — the pipeline config loaded by `run_gradio_app.py`.
- `configs/default.yaml` — base experiment config (reference; carried from disorder_fix).
- Schemas + validation: `src/config/schema.py` (pydantic v2), `src/utils/config.py`.

## Tests

`tests/` mirrors `src/`. `tests/conftest.py` globally mocks the slow MFA aligner and
provides fixtures. GPU models are mocked at the unit level. This suite is the
correctness baseline for parity-gated optimization.

## Not present (excluded from disorder_fix on purpose)

Training, labeling, mining, analysis, evaluation, benchmarks/stress_test, W&B, and
the U-Net/diffusion/HiFi-GAN inpainting stack (`models/`, `pipelines/fft.py`,
`hifi_gan.py`, `vocoder.py`, `inference_pipeline.py`, `inference/infer.py`).
