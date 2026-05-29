# Real-time optimization plan

The pipeline works (see `disorder_fix`); it is just slow. This document is the plan
to make it real-time, and the running record of what was tried and measured. It is
governed by the constitution — especially **Numerical Parity Under Optimization**
(Principle I) and **Latency & Throughput Budgets** (Principle II).

## Where the time goes (per request)

| Stage | Module | Device | Cost (rough, PyTorch baseline) | Optimizable? |
|-------|--------|--------|-------------------------------|--------------|
| ASR | `preprocessing/asr.py` (GigaAM-v3 RNN-T) | GPU | hundreds of ms | **Yes — TensorRT** |
| Align (gigaam) | `preprocessing/gigaam_aligner.py` | GPU | ~150 ms (reuses ASR encoder) | folds into ASR |
| Align (mfa) | `preprocessing/forced_aligner.py` | CPU + subprocess | ~150 s/file | avoid on hot path |
| Hard-S filter | `preprocessing/hard_s_filter.py` | CPU | negligible | no |
| TTS | `inference/omnivoice_adapter.py` (OmniVoice) | GPU | the dominant cost | **Yes — TensorRT + fewer steps** |
| Splice | `inference/word_splice.py` | CPU | negligible | no |

Two GPU models dominate: **GigaAM-v3** and **OmniVoice**. Everything else is light
CPU glue. Optimize the GPU models; leave the glue alone.

## Step 0 — Baseline + parity harness (do this first)

Nothing below is allowed to land without this.

1. Pick a fixed fixture set (start with `data/test/khovansky/`).
2. Run the unoptimized PyTorch pipeline and persist **reference outputs**:
   - ASR transcript text,
   - alignment word boundaries (start/end per word),
   - the corrected-audio waveform.
3. Record **per-stage latency** (p50/p95) and GPU memory on the target hardware.
4. Write a parity check that, given a candidate build, asserts:
   - ASR text exact-match (or WER ≤ ε),
   - word boundaries within N ms,
   - corrected audio within an SNR / log-spectral-distance bound.

The pipeline already exposes a `StageProbe` hook
(`ResynthesisPipeline(..., probe=...)`) — reuse it for the latency capture instead
of adding ad-hoc timers.

## Step 1 — Cheap wins, no export

- **Default to `aligner: "gigaam"`** so alignment reuses the ASR encoder (no second
  model, ~150 ms vs ~150 s for MFA). MFA stays available for offline phoneme work.
- **Pay model load once**: load OmniVoice + GigaAM at startup and warm them up before
  serving (already true for the long-lived Gradio runner; make it explicit for any
  server).
- **Sweep `omnivoice.num_step`** under parity — fewer diffusion steps is the single
  biggest TTS latency lever. Find the smallest step count that still passes parity.
- **fp16 everywhere it holds** (already the OmniVoice default `dtype: float16`);
  confirm GigaAM runs fp16 under parity.

## Step 2 — TensorRT engines

For each GPU model, export and gate on parity:

- **GigaAM-v3 encoder** → ONNX → TensorRT. The RNN-T decoder/joint loop is
  autoregressive and small; keep it in PyTorch or export as a separate engine. The
  greedy decode (and the timed-token loop in `transcribe_with_timings`) is where the
  word-boundary timing comes from — parity on boundaries matters here.
- **OmniVoice** → the diffusion/codec backbone is the heavy part. Export the
  transformer to TensorRT; the iterative MaskGIT-style decode stays orchestrated in
  Python. fp16 first; int8 only where parity survives.

Engines are **device-specific** — never commit `.engine`/`.plan` files (gitignored).
Build them on the target GPU from committed ONNX or build scripts.

## Step 3 — Triton Inference Server

- Stand up a `model_repository/` with the TensorRT engines as Triton models.
- Configure **instance groups** (GPU residency), **dynamic batching**, and
  **warmup** so first-request latency is not mistaken for steady state.
- Re-point the adapters (`omnivoice_adapter.py`, `asr.py`) at Triton via
  `tritonclient` — selected by config, behind the existing adapter interface, so the
  orchestrator (`ResynthesisPipeline`) does not change.
- Verify end-to-end parity again (the seams between Triton stages can introduce
  dtype/layout drift).

## Step 4 — Further latency tricks (as needed)

- **Phoneme-level splice** instead of word-level (less TTS audio synthesized per fix)
  — see the P3 note carried in the research repo's `todo_omnivoice_pipeline.md`.
- **Speech-edit / partial-mask infill** in OmniVoice (synthesize only the disordered
  region) — the strongest technical direction; larger effort.
- CUDA graphs / kernel fusion for the autoregressive RNN-T decode if it becomes the
  bottleneck after the backbone is on TensorRT.

## Recording results

Each optimization gets an entry: what changed, parity result (pass/fail + tolerance),
and before/after p50/p95/RTF/GPU-mem on named hardware. Keep raw benchmark artifacts
under `benchmarks/` (gitignored results, committed scripts + summary).

| Date | Change | Parity | p50 before→after | RTF before→after | Kept? |
|------|--------|--------|------------------|------------------|-------|
| _tbd_ | baseline capture | — | — | — | — |
