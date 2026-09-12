"""
Same as before, but pushes the tunnel URL to ntfy.sh the instant it's
found -- bypasses Kaggle's unreliable mid-run log serving entirely.
"""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

COMFY_DIR = Path("/tmp/ComfyUI")
MODELS_DIR = COMFY_DIR / "models"
NTFY_TOPIC = "wan22studio-deepfake807-notify"  # unique-ish topic name

def log(msg): print(f"[worker] {msg}", flush=True)

def run_step(name, cmd, **kwargs):
    log(f"START: {name}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    dt = round(time.time() - t0, 1)
    if result.returncode != 0:
        log(f"FAILED ({dt}s): {name}")
        log(f"stderr: {result.stderr[-1500:]}")
        raise RuntimeError(name)
    log(f"DONE ({dt}s): {name}")

CUSTOM_NODES = {
    "ComfyUI-GGUF": "https://github.com/city96/ComfyUI-GGUF.git",
    "ComfyUI-WanAnimatePreprocess": "https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git",
    "ComfyUI-WanVideoWrapper": "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
    "ComfyUI-segment-anything-2": "https://github.com/kijai/ComfyUI-segment-anything-2.git",
    "comfyui_controlnet_aux": "https://github.com/Fannovel16/comfyui_controlnet_aux.git",
    "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
}

MODEL_FILES = [
    ("QuantStack/Wan2.2-Animate-14B-GGUF", "Wan2.2-Animate-14B-Q2_K.gguf", "diffusion_models", "diffusion model"),
    ("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q5_K_M.gguf", "text_encoders", "text encoder"),
    ("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", "vae", "VAE"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision", "CLIP vision"),
]

def notify(message: str):
    try:
        subprocess.run(["curl", "-s", "-d", message, f"https://ntfy.sh/{NTFY_TOPIC}"],
                        capture_output=True, timeout=15)
        log(f"notified: {message}")
    except Exception as e:
        log(f"notify failed: {e}")

def main():
    import torch
    log(f"gpu_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if not torch.cuda.is_available():
        sys.exit(1)

    notify("Wan2.2 worker: starting setup...")

    run_step("clone ComfyUI",
              ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_DIR)])
    run_step("pip install ComfyUI requirements",
              [sys.executable, "-m", "pip", "install", "-q", "-r", str(COMFY_DIR / "requirements.txt")])
    run_step("pip install hf_xet", [sys.executable, "-m", "pip", "install", "-q", "hf_xet"])

    nodes_dir = COMFY_DIR / "custom_nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    for name, url in CUSTOM_NODES.items():
        target = nodes_dir / name
        run_step(f"clone {name}", ["git", "clone", "--depth", "1", url, str(target)])
        req = target / "requirements.txt"
        if req.exists():
            run_step(f"pip install requirements for {name}",
                      [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])

    from huggingface_hub import hf_hub_download
    for repo, filename, subfolder, label in MODEL_FILES:
        dest_dir = MODELS_DIR / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / Path(filename).name
        path = hf_hub_download(repo_id=repo, filename=filename, token=os.environ.get("HF_TOKEN") or None)
        shutil.copy(path, dest_path)
        log(f"DONE: {label}")

    workflows_dir = COMFY_DIR / "user" / "default" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    run_step("download WanAnimate example workflow",
              ["curl", "-sL", "-o", str(workflows_dir / "wanvideo_WanAnimate_example_01.json"),
               "https://raw.githubusercontent.com/kijai/ComfyUI-WanVideoWrapper/main/example_workflows/wanvideo_WanAnimate_example_01.json"])

    run_step("download cloudflared",
              ["wget", "-q", "-O", "/tmp/cloudflared",
               "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"])
    subprocess.run(["chmod", "+x", "/tmp/cloudflared"], check=True)

    log("starting ComfyUI")
    comfy_proc = subprocess.Popen([sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188", "--enable-cors-header"], cwd=str(COMFY_DIR))
    time.sleep(25)
    if comfy_proc.poll() is not None:
        notify("Wan2.2 worker: FAILED - ComfyUI crashed on startup")
        sys.exit(1)

    with open("/tmp/cloudflared.log", "w") as f:
        tunnel_proc = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://localhost:8188"],
                                        stdout=f, stderr=subprocess.STDOUT)
    time.sleep(12)

    with open("/tmp/cloudflared.log") as f:
        content = f.read()
    m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", content)
    if m:
        url = m.group(0)
        log(f"TUNNEL_URL_FOUND: {url}")
        notify(f"Wan2.2 ComfyUI READY: {url}")
    else:
        log("TUNNEL_URL_NOT_FOUND")
        notify("Wan2.2 worker: FAILED - no tunnel URL found")
        sys.exit(1)

    log("Sleeping 25 minutes for manual testing.")
    time.sleep(25 * 60)
    notify("Wan2.2 worker: session time up, shutting down")
    tunnel_proc.terminate()
    comfy_proc.terminate()

if __name__ == "__main__":
    main()
