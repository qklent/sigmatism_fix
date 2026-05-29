# RunPod dev runbook

Repeatable spin-up / operate / teardown for the **single-pod, two-process** dev
setup (Triton + app co-located later; just the PyTorch app for now). Designed to be
driven via the RunPod MCP where possible, with SSH for everything the MCP can't do
(logs, shell, running services).

## Topology

```
one RunPod GPU pod
  ├─ app process (Gradio + ResynthesisPipeline)   :7860  → RunPod HTTP proxy (browser)
  └─ (later) Triton server  :8000/8001  localhost  ·  :8002 metrics (curl over SSH)
  SSH :22 (tcp)  → tail logs, run benchmarks, nvidia-smi
```

## Prerequisites (one-time, already done for this account)

- **RunPod account secrets** (referenced as `{{ RUNPOD_SECRET_* }}`): `hf_token`,
  `github_token`. These are injected as `HF_TOKEN` / `GITHUB_TOKEN` env on the pod.
- **SSH key** `~/.ssh/runpod` (ED25519) registered in RunPod account settings →
  auto-added to the pod's `authorized_keys`.
- GPU choice: **RTX 4090 24 GB** (fits OmniVoice + GigaAM; baseline work is
  GPU-agnostic so any available 24 GB card — A5000 / 3090 / L4 — is an acceptable
  substitute).

## 1. Spin up the pod

Via the MCP (`create-pod`). Parameters we use:

| field | value |
|-------|-------|
| `name` | `sigmatism-dev` |
| `imageName` | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| `cloudType` | `COMMUNITY` (cheapest 4090; `SECURE` for more reliable stop/start) |
| `gpuTypeIds` | `["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"]` |
| `containerDiskInGb` | `30` |
| `volumeInGb` | `60` · mount `/workspace` |
| `ports` | `["22/tcp", "7860/http", "8888/http"]` |
| `env` | `HF_TOKEN={{RUNPOD_SECRET_hf_token}}`, `GITHUB_TOKEN={{RUNPOD_SECRET_github_token}}`, `HF_HOME=/workspace/.cache/huggingface` |

> **Capacity caveat.** This MCP's `create-pod` can't query live availability or
> target a specific machine — it resolves to one machine and fails with
> *"This machine does not have the resources"* / *"no instances currently available"*
> when that card is full. When 4090 capacity is tight:
> 1. retry (capacity fluctuates), and/or widen `gpuTypeIds`;
> 2. **fall back to the RunPod web console**, which shows live availability and lets
>    you pick an available machine. Once it exists, manage it via the MCP
>    (`get-pod` / `stop-pod` / `delete-pod`) and operate it over SSH exactly as below.

## 2. Connect

Get connection info: MCP `get-pod` with `includeMachine: true` → public IP + the
mapped external port for `22/tcp`.

```bash
ssh root@<POD_IP> -p <SSH_PORT> -i ~/.ssh/runpod
# or RunPod's basic proxy:  ssh <POD_ID>@ssh.runpod.io -i ~/.ssh/runpod
```

## 3. Bootstrap (on the pod, first boot or after a stop)

```bash
# GITHUB_TOKEN / HF_TOKEN / HF_HOME are already in the pod env.
cd /workspace && \
  ( [ -d sigmatism_fix ] || git clone "https://oauth2:${GITHUB_TOKEN}@github.com/qklent/sigmatism_fix.git" ) && \
  bash sigmatism_fix/deploy/bootstrap_pod.sh
```

Installs miniforge + the `sigmatism` conda env + repo **under `/workspace`** so they
survive stop/start (the container disk is wiped on stop). HF cache → `/workspace`.

## 4. Start services

```bash
bash /workspace/sigmatism_fix/deploy/start_services.sh
```

Launches the Gradio app in tmux with logs at `/workspace/logs/app.log`. Open the
UI at `https://<POD_ID>-7860.proxy.runpod.net` (first request downloads the models).

## 5. Observe (over SSH)

```bash
tail -f /workspace/logs/app.log          # app + pipeline stage logs
tmux attach -t app                        # live console
nvidia-smi                                # GPU/VRAM
# later, with Triton up:
curl -s localhost:8002/metrics | grep -E 'nv_inference_(count|compute|queue)'
```

## 6. Teardown

- **Stop** (`stop-pod`): GPU billing stops; `/workspace` is preserved → fast resume
  via steps 2 → 4 (skip bootstrap if the env is intact). You still pay for the
  `/workspace` disk while stopped.
- **Delete** (`delete-pod`): everything goes, including `/workspace`. Use when done
  for a while. Re-create with step 1.

## Persistence model (no network volume — MCP can't attach one)

| Lives under | Survives stop? | Survives delete? |
|-------------|----------------|------------------|
| `/workspace` (miniforge, repo, HF cache, engines, logs) | ✅ | ❌ |
| `/`, `/root` (apt packages, `~/.mfa`) | ❌ (re-run bootstrap) | ❌ |

For pod-*independent* persistence (survives delete, movable across machines), create
a **network volume** in the console and attach it at pod creation from the console.
Worth doing once TensorRT engines become expensive to rebuild.

## Cost notes

- RTX 4090 community ≈ $0.35–0.70/hr while running; **stop** when idle.
- `/workspace` disk ≈ $0.10/GB/mo whether running or stopped (60 GB ≈ $6/mo) — delete
  the pod to stop this too.
