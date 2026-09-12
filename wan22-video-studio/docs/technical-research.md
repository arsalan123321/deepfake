# Technical Research (Phase 1)

## Model choice
Using Wan2.2 Animate (original, Mix/Move modes) — NOT the newer "Wan Animate 2"
nodes. Original is better documented and has proven low-VRAM community workflows.

- Mix mode = character replacement (swap person in source video with reference image)
- Move mode = animation (reference image driven by motion from a video)

## Required files (GGUF / low-VRAM path for T4)
- Diffusion model: Wan2.2 Animate 14B, GGUF quant (Q4_K_M or Q5_K_M, ~10-12GB)
  from QuantStack/city96-style GGUF repacks
- Text encoder: umt5-xxl-encoder GGUF
- VAE: wan_2.1_vae.safetensors (~small, keep in fp16/bf16, not worth quantizing)
- CLIP Vision: clip_vision_h.safetensors
- LoRA: lightx2v step-distillation LoRA (cuts sampling steps significantly)

## Required custom nodes
- ComfyUI-GGUF (city96) — GGUF unet/clip loaders
- ComfyUI-WanAnimatePreprocess
- ComfyUI-segment-anything-2 (SAM2, for auto masking in Mix mode)
- comfyui_controlnet_aux (DWPose estimator for pose extraction)
- ComfyUI-VideoHelperSuite (VHS_LoadVideo, VHS_VideoInfo, video I/O)
- KJNodes (Kijai) — misc helper nodes used in reference workflows

## VRAM reality on T4 (16GB)
Full precision 14B: 40-80GB — does NOT fit.
FP8: ~22-26GB — does NOT fit on single T4.
GGUF Q4-Q5 + offload: ~10-14GB resident — fits, with headroom for VAE/latents
at 480x832 and short clips (<=81 frames). This is our target profile.

## T4x2 reality
No tensor-parallel sharding of one DiT across 2 GPUs in vanilla ComfyUI.
Realistic use of 2nd GPU: run text encoder/CLIP vision/VAE there, keep DiT on GPU0.
Do not claim T4x2 unlocks 720p until empirically tested on the actual worker.

## Kaggle automation reality
- Start: `kaggle kernels push --accelerator NvidiaTeslaT4` (or NvidiaTeslaT4Highmem)
- Poll: `kaggle kernels status <user>/<kernel-id>`
- Retrieve output: `kaggle kernels output <user>/<kernel-id> -p <dest>`
- No literal remote "stop" endpoint — the kernel must exit on its own
  (we implement this via a polling flag file the notebook checks between jobs)
- No exact quota-remaining API — track locally, label as estimate
