# Wan2.2 Video Studio

Personal AI video character-replacement/animation tool.
Pipeline: Browser → FastAPI backend → Kaggle T4 GPU → ComfyUI (headless) → Wan2.2 Animate → output.mp4 → back to browser.

**Status: Phase 1-2 (Kaggle worker bring-up). No frontend yet.**

## Honest limitations (read this before opening an issue at 2am)

- Wan2.2 Animate 14B at full precision needs ~40-80GB VRAM. A Kaggle T4 has 16GB.
  We run a **GGUF-quantized** version (~10-12GB) instead. Expect lower quality
  and slower generation than a full-precision cloud GPU.
- Kaggle has no "start/stop a VM" API. "Starting the GPU" means pushing a
  kernel run via the Kaggle API and polling it. "Stopping" means the kernel
  finishing or being told to wrap up — there's no hard remote kill switch.
- Kaggle does not expose exact remaining GPU-hour quota via API. We track
  session time locally and label it as an estimate.
- T4x2 does not mean double VRAM for one model. ComfyUI doesn't shard a
  single model across GPUs; at best we put the text encoder/VAE on GPU2.

## Requirements

- Kaggle account with phone verification (required for GPU kernels) and an API token
  (kaggle.com/settings → Create New Token)
- Optional: Hugging Face token if any model repo requires auth
- Python 3.10+, Node 18+ (once frontend exists)

## Repo layout

- `kaggle/notebook/` — the worker script that runs ON Kaggle
- `kaggle/comfy/workflows/` — ComfyUI workflow JSON templates
- `backend/` — FastAPI app (Phase 3+)
- `frontend/` — web UI (Phase 6+)
- `docs/` — architecture, research, troubleshooting

## Setup — Kaggle worker (current phase)

See `docs/kaggle.md`.
