# sigmatism_fix

**Real-time hard-S (sigmatism) speech correction.** Takes disordered Russian
speech with a "hard" or lisped С/Ш/Ж/etc., and resynthesizes the affected words
in a clean voice — then splices them back into the original audio so prosody and
timing are preserved.

This repo is the **optimization / serving** sibling of the research repo
[`disorder_fix`](https://github.com/qklent/disorder_fix), where the pipeline was
designed and validated. Here the goal is narrow and concrete:

> Make the working pipeline fast enough for real-time use — by hosting **OmniVoice**
> (TTS) and **GigaAM-v3** (ASR + aligner) on GPU via **Triton Inference Server**
> with **TensorRT**, plus other latency tricks — **without changing what it outputs.**

## What the pipeline does

```
 input speech (disordered)            reference voice (clean, sibilant-rich)
        │                                        │
        ▼                                        │
  ┌───────────────┐   GigaAM-v3                  │
  │  ASR (text)   │   (skipped if transcript     │
  └───────────────┘    provided)                 │
        │                                        │
        ▼                                        │
  ┌───────────────┐   GigaAM RNN-T timesteps (default)
  │  Word align   │   or MFA phoneme align
  └───────────────┘
        │
        ▼
  ┌───────────────┐   phoneme-level (MFA) or espeak text-level (gigaam)
  │ hard-S filter │   → which words contain a hard /s/, /sʲ/, …
  └───────────────┘
        │  (target words)
        ▼
  ┌───────────────┐   OmniVoice zero-shot voice cloning
  │  TTS resynth  │ ◄─── reference voice
  └───────────────┘
        │
        ▼
  ┌───────────────┐   align resynth → match words → Hann-crossfade splice
  │  word splice  │   only the hard-S words are replaced
  └───────────────┘
        │
        ▼
  corrected audio (original prosody + timing, clean sibilants)
```

Orchestrated by `src/pipelines/resynthesis_pipeline.py::ResynthesisPipeline`.
The whole thing is **config-driven** — see `configs/resynthesis.yaml`.

## Models

| Role            | Model                     | Source (HF Hub)                    |
|-----------------|---------------------------|------------------------------------|
| TTS (resynth)   | OmniVoice                 | `k2-fsa/OmniVoice`                 |
| ASR + aligner   | GigaAM-v3 (RNN-T)         | `ai-sage/GigaAM-v3` (`e2e_rnnt`)   |
| Phoneme aligner | Montreal Forced Aligner   | `russian_mfa` (optional backend)  |

Weights are downloaded from HuggingFace Hub on first use (cached under
`~/.cache/huggingface`). These two GPU models — OmniVoice and GigaAM-v3 — are the
primary targets for TensorRT/Triton optimization.

## Quick start

This needs a CUDA GPU. The easiest path is a fresh pod via `setup_pod.sh`
(installs system deps, conda env, MFA models). Manually:

```bash
# 1. System deps (espeak-ng is required for the hard-S text detection)
sudo apt-get install -y espeak-ng ffmpeg libsndfile1

# 2. Python env (conda recommended so MFA can be installed too)
conda env create -f environment.yml     # creates the `sigmatism` env
conda activate sigmatism
# (or, pip only — no MFA backend:)
#   pip install -e ".[dev,inference,resynthesis,app]"

# 3. Credentials
cp .env.example .env        # add HF_TOKEN

# 4. Run the Gradio demo (reference clip + speech-to-fix → corrected audio)
HF_TOKEN=... python scripts/run_gradio_app.py
# open http://localhost:7860  (SSH-forward from a pod: ssh -L 7860:localhost:7860 ...)
```

### Run the tests (CPU, no GPU needed)

```bash
pip install -e ".[dev,inference,app]"
pytest tests/ -q
```

The slow MFA aligner and the GPU models are mocked in `tests/conftest.py`, so the
unit suite runs on CPU in seconds. This suite is the **correctness baseline** that
every optimization must preserve (see the constitution, Principle I).

## Optimization roadmap

The serving/optimization plan lives in
[`docs/realtime_optimization.md`](docs/realtime_optimization.md). In short:

1. Profile the PyTorch baseline; record per-stage latency budgets.
2. Capture reference outputs for parity gating.
3. Export GigaAM-v3 encoder and OmniVoice to ONNX → TensorRT engines.
4. Serve both behind Triton (instance groups, dynamic batching, warmup).
5. Re-point the adapters at Triton; verify numerical parity + measure speedup.

## Directory layout

| Path | Purpose |
|------|---------|
| `src/pipelines/resynthesis_pipeline.py` | End-to-end orchestration |
| `src/inference/omnivoice_adapter.py` | OmniVoice TTS wrapper |
| `src/inference/word_splice.py` | Hann-crossfade word-level splicing |
| `src/preprocessing/asr.py` | GigaAM-v3 ASR (+ token timings) |
| `src/preprocessing/gigaam_aligner.py` | GigaAM RNN-T word alignment |
| `src/preprocessing/forced_aligner.py` | MFA phoneme alignment (optional) |
| `src/preprocessing/hard_s_filter.py` | Hard-S word detection |
| `src/preprocessing/phonemizer.py` | espeak-ng Russian phonemizer |
| `src/data/audio_adapter.py` | Audio I/O + mel adapter |
| `src/config/schema.py` | Pydantic config models |
| `src/apps/gradio_app/` | Single-user Gradio demo UI |
| `scripts/run_gradio_app.py` | Demo launcher |
| `configs/` | `resynthesis.yaml` (pipeline) + `default.yaml` |
| `tests/` | CPU unit tests mirroring `src/` |
| `docs/` | Architecture diagrams + optimization plan |
| `data/test/` | Small audio fixtures for smoke/latency benchmarking |
| `.specify/` + `.claude/commands/speckit.*` | Spec-driven dev (speckit) |

## Configuration

`configs/resynthesis.yaml` drives the pipeline:

```yaml
device: "cuda:0"
omnivoice:
  model_name: "k2-fsa/OmniVoice"
  dtype: "float16"
  num_step: 32          # OmniVoice diffusion steps — a key latency/quality knob
  speed: 1.0
  sample_rate: 24000
resynthesis:
  aligner: "gigaam"     # "gigaam" (fast, RNN-T) or "mfa" (slow, phoneme-level)
  crossfade_ms: 10.0
  guard_ms: 80.0        # ~80 for GigaAM RNN-T late emission; ~0 for MFA
  provide_ref_text: true
```

| Field | Why it matters for real-time |
|-------|------------------------------|
| `aligner` | `gigaam` reuses the ASR model (~150 ms); `mfa` is ~150 s/file |
| `omnivoice.num_step` | Fewer diffusion steps = faster TTS, lower quality |
| `omnivoice.dtype` | `float16` baseline; candidate for int8 under parity |

## Status

Skeleton copied from the validated `disorder_fix` pipeline. No optimization landed
yet — the next step is profiling + a parity harness. See `CHANGELOG.md`.

## License

MIT — see [LICENSE](LICENSE).
