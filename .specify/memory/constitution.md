<!--
SYNC IMPACT REPORT
==================
Version change: (new repo) → 1.0.0
Origin: Adapted from disorder_fix Constitution v1.0.0. This repo's focus shifted
from training/experimentation to REAL-TIME INFERENCE SERVING of a fixed pipeline
(OmniVoice TTS + GigaAM-v3 ASR/aligner) on GPU via Triton + TensorRT.

Carried over (universal engineering principles):
  - Config-Driven Everything
  - Modular Architecture & Component Isolation
  - Reproducibility & Determinism
  - Audio & I/O Adapter Standards
  - Testing
  - Security & Credentials
  - Documentation & Tracking

Replaced (training-specific → serving-specific):
  - W&B experiment tracking / checkpointing / sanity-overfit / dataset splits
    → Numerical Parity Under Optimization
    → Latency & Throughput Budgets
    → GPU Serving Discipline

Templates checked:
  - .specify/templates/plan-template.md   ✅ Constitution Check section derives from these principles
  - .specify/templates/spec-template.md   ✅ compatible
  - .specify/templates/tasks-template.md  ✅ compatible
-->

# sigmatism_fix Constitution

The mission of this repository is to take a **working but slow** hard-S correction
pipeline and make it **fast enough for real-time use** — without changing what it
outputs. Every principle below serves that mission: preserve behavior, measure
everything, optimize deliberately.

## Core Principles

### I. Numerical Parity Under Optimization (NON-NEGOTIABLE)

Optimization MUST NOT silently change model behavior.

- Before optimizing any stage, capture a **reference output** from the unoptimized
  PyTorch baseline (transcript text, alignment word boundaries, and corrected-audio
  waveform) for a fixed fixture set.
- Every optimization (TensorRT engine, ONNX export, quantization, fused kernel,
  batching change) MUST be gated by a **parity check** against that baseline within
  a declared tolerance (e.g. ASR text exact-match or WER ≤ threshold; alignment
  boundaries within N ms; audio within an SNR / spectral-distance bound).
- A change that fails parity is rejected or its tolerance is explicitly justified
  and recorded — never waved through.
- Quantization / lower precision (fp16, int8) is allowed only where parity holds.

**Rationale**: The whole value of this pipeline is the quality of the correction.
A 3× speedup that degrades the /s/ correction is a regression, not a win. Parity
gates make "faster" and "still correct" inseparable.

### II. Latency & Throughput Budgets

Performance is a first-class, measured requirement — not a vibe.

- Each pipeline stage (ASR, alignment, hard-S filter, TTS, splice) has a **measured
  latency budget**. The end-to-end target (p50/p95 and real-time factor) is declared
  in config or a benchmark spec.
- No optimization is merged without **before/after numbers** on the same hardware:
  p50, p95, RTF, and GPU memory. "It feels faster" is not evidence.
- Benchmarks are reproducible artifacts: fixed inputs, recorded hardware/driver/
  library versions, results checked in under `benchmarks/`.
- Regressions in latency or memory are treated like correctness bugs.

**Rationale**: Real-time is a number, not an adjective. Without per-stage budgets
you cannot tell which stage to optimize next or whether a change actually helped.

### III. Config-Driven Everything

- All model names, revisions, devices, precision/dtype, aligner backend, sample
  rates, and serving parameters (batch size, concurrency, num_step) MUST live in
  config (YAML), never hardcoded in source.
- Device/precision selection is a config field. Strings like `"cuda:0"` MUST NOT be
  buried in source logic — they flow from config.
- Each run/benchmark MUST be able to snapshot the exact config it used.

**Rationale**: Reproducible performance and reproducible behavior both require the
full run to be described by a committed, swappable config.

### IV. Modular Architecture & Component Isolation

- The ASR, aligner, TTS, hard-S filter, and splice stages MUST remain separate,
  swappable components behind stable interfaces (the existing adapter classes).
- A backend swap (PyTorch → TensorRT → Triton-remote) for any single stage MUST be
  possible by changing config / an adapter, not by editing the orchestration code.
- Orchestration (`ResynthesisPipeline`) stays backend-agnostic: it calls adapters,
  it does not know whether a stage runs locally in PyTorch or remotely on Triton.

**Rationale**: Optimization proceeds one stage at a time. Tight coupling would force
all-or-nothing rewrites and make A/B comparison of backends impossible.

### V. Audio & I/O Adapter Standards

- Audio I/O and feature extraction MUST stay behind the thin adapter layer
  (`src/data/audio_adapter.py`) so backends (torchaudio, soundfile, librosa,
  GPU resamplers) can be swapped in one place.
- Raw input audio MUST never be mutated in place. Per-run artifacts are written to
  dedicated run directories.
- Sample-rate assumptions (16 kHz ASR, 24 kHz OmniVoice, native input SR) MUST be
  explicit and centralized, not re-derived ad hoc.

**Rationale**: Sample-rate and format mismatches are a classic silent source of both
quality loss and wasted latency (needless resampling).

### VI. GPU Serving Discipline

- Model load, warmup, and resident-memory cost MUST be explicit and paid once at
  startup — never per request on the hot path.
- Concurrency, batching, and request serialization MUST be explicit (the Gradio
  runner already single-flights; a Triton deployment declares instance groups and
  dynamic batching in its model config).
- Warmup MUST run before a server reports ready, so first-request latency is not
  mistaken for steady-state latency in benchmarks.
- GPU OOM and cold-start behavior are part of the contract and MUST be tested.

**Rationale**: Most "real-time" failures are not throughput — they are cold starts,
per-request model loads, and unbounded concurrency. Make them explicit.

### VII. Testing

- All pure-Python pipeline logic (filtering, splicing, alignment parsing, config
  validation, persistence) MUST have unit tests that run on CPU.
- Heavy/slow/networked dependencies (MFA, GPU models) are mocked in unit tests
  (see `tests/conftest.py`); real-model paths run as explicitly-marked tests.
- Optimization work MUST add a parity test (Principle I) and, where feasible, a
  latency micro-benchmark (Principle II).
- Test files live under `tests/` mirroring `src/`.

**Rationale**: A correctness baseline is the prerequisite for safe optimization.
You cannot preserve behavior you cannot test.

### VIII. Security & Credentials

- HuggingFace tokens and any credentials MUST come from environment variables or a
  `.env` file. `.env` is gitignored; `.env.example` documents the keys.
- No credentials, tokens, or machine-specific absolute paths in committed configs.
- A pre-commit hook SHOULD block accidental secret commits.

**Rationale**: One leaked token compromises model access and cloud accounts.

### IX. Documentation & Tracking

- `codemap.md` is a living architecture map and MUST be updated when structure changes.
- `CHANGELOG.md` follows Keep a Changelog and is updated with every significant change.
- Benchmark results and optimization decisions (what was tried, the numbers, what was
  kept) MUST be recorded so the performance history is auditable.
- Spec-driven features live under `specs/NNN-<slug>/` via the speckit workflow.

**Rationale**: Optimization is iterative and full of dead ends. Without a record of
what was tried and measured, the same experiments get re-run.

## Engineering Standards

### Dependencies & Environment

- Runtime dependency versions are pinned/bounded in `pyproject.toml`. PyTorch /
  TensorRT / Triton version coordination is documented (these are tightly coupled).
- `tensorrt` and `torch-tensorrt` are sourced from the NVIDIA NGC container rather
  than pinned blindly in `pyproject.toml`.
- Notebooks (if any) stay in `notebooks/` for exploration only — never on the
  serving hot path.

### Code Quality Gates

Every change MUST pass before merge:

1. Unit tests pass on CPU.
2. Lint/format pass (`ruff`).
3. No hardcoded device strings, credentials, or absolute paths.
4. Any optimization includes a parity result (Principle I) and before/after latency
   numbers (Principle II).
5. `codemap.md` / `CHANGELOG.md` updated if structure changed.

## Governance

This constitution supersedes other conventions; where conflicts arise, it is
authoritative.

**Amendment procedure**:

1. Propose the amendment in a PR with rationale.
2. Bump the version per semantic versioning:
   - MAJOR: backward-incompatible removal/redefinition of a principle.
   - MINOR: new principle/section or material expansion.
   - PATCH: clarifications and wording.
3. Update `Last Amended` and `Version`.
4. Propagate to affected `.specify/templates/`.
5. Add a `CHANGELOG.md` entry.

**Compliance**: Justified exceptions MUST be documented inline with a
`# CONSTITUTION_EXCEPTION:` comment explaining the reason.

**Version**: 1.0.0 | **Ratified**: 2026-05-29 | **Last Amended**: 2026-05-29
