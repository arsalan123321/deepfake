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
    ("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors", "text_encoders", "text encoder (bf16, required by TextEncode Cached node)"),
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
    wf_path = workflows_dir / "wanvideo_WanAnimate_example_01.json"
    run_step("download WanAnimate example workflow",
              ["curl", "-sL", "-o", str(wf_path),
               "https://raw.githubusercontent.com/kijai/ComfyUI-WanVideoWrapper/main/example_workflows/wanvideo_WanAnimate_example_01.json"])

    log("patching workflow: fixing known filenames and bypassing optional LoRA nodes")
    wf_text = wf_path.read_text()
    FILENAME_FIXES = {
        "WanVideo\\2_2\\Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors": "Wan2.2-Animate-14B-Q2_K.gguf",
        "umt5-xxl-enc-bf16.safetensors": "umt5-xxl-enc-bf16.safetensors",  # already correct filename, just ensuring path match
        "wanvideo\\Wan2_1_VAE_bf16.safetensors": "Wan2_1_VAE_bf16.safetensors",
    }
    for old_name, new_name in FILENAME_FIXES.items():
        wf_text = wf_text.replace(old_name, new_name)

    import json as _json
    try:
        wf_json = _json.loads(wf_text)
        bypassed = 0
        for node in wf_json.get("nodes", []):
            if node.get("type") == "WanVideo Lora Select Multi":
                node["mode"] = 4  # 4 = bypass in ComfyUI
                bypassed += 1
        wf_text = _json.dumps(wf_json)
        log(f"bypassed {bypassed} LoRA Select Multi node(s) automatically")
    except Exception as e:
        log(f"WARNING: could not parse workflow JSON to bypass LoRA nodes: {e}")

    wf_path.write_text(wf_text)
    log("workflow patched and saved")

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

    log("Waiting for stop signal or 25 minute timeout, checking every 15s.")
    import urllib.request as _ur
    stop_topic = "wan22studio-deepfake807-stopsignal"
    deadline = time.time() + 25 * 60
    stopped_by_signal = False
    while time.time() < deadline:
        time.sleep(15)
        try:
            req = _ur.Request(f"https://ntfy.sh/{stop_topic}/json?poll=1&since=30s")
            with _ur.urlopen(req, timeout=8) as resp:
                lines = resp.read().decode().strip().split("\n")
            if any(line.strip() for line in lines):
                log("STOP SIGNAL RECEIVED -- shutting down now")
                stopped_by_signal = True
                break
        except Exception as e:
            log(f"stop-signal check failed (non-fatal): {e}")
    if not stopped_by_signal:
        log("25 minute timeout reached")
    notify("Wan2.2 worker: shutting down")
    tunnel_proc.terminate()
    comfy_proc.terminate()

if __name__ == "__main__":
    main()
