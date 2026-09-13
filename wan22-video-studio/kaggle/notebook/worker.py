"""
Wan2.2 Video Studio — Kaggle worker bootstrap (full pipeline).
Auto-patches the WanAnimate workflow using JSON parsing + substring
matching on widget values (robust against path/escaping differences),
instead of fragile raw-text find/replace.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

COMFY_DIR = Path("/tmp/ComfyUI")
MODELS_DIR = COMFY_DIR / "models"


def log(msg: str) -> None:
    print(f"[worker] {msg}", flush=True)


def detect_hardware() -> dict:
    info = {"gpu_count": 0, "gpus": [], "cuda_available": False,
            "ram_gb": None, "disk_free_gb": None, "profile": "UNKNOWN"}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            for i in range(info["gpu_count"]):
                p = torch.cuda.get_device_properties(i)
                info["gpus"].append({"index": i, "name": p.name,
                                      "vram_gb": round(p.total_memory / (1024**3), 1)})
    except Exception as e:
        log(f"torch detection failed: {e}")
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        pass
    try:
        _, _, free = shutil.disk_usage("/tmp")
        info["disk_free_gb"] = round(free / (1024**3), 1)
    except Exception:
        pass
    if info["gpu_count"] == 0:
        info["profile"] = "NO_GPU"
    elif info["gpu_count"] == 1:
        info["profile"] = "LOW_VRAM_T4" if info["gpus"][0]["vram_gb"] <= 16 else "SINGLE_HIGH_VRAM"
    else:
        vrams = [g["vram_gb"] for g in info["gpus"]]
        info["profile"] = "T4_DUAL_LOW_VRAM" if all(v <= 16 for v in vrams) else "MULTI_HIGH_VRAM"
    return info


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
    ("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors", "text_encoders", "text encoder"),
    ("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", "vae", "VAE"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision", "CLIP vision"),
]

VALUE_FIXES = {
    "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors": "Wan2.2-Animate-14B-Q2_K.gguf",
}
VAE_CORRECT = "Wan2_1_VAE_bf16.safetensors"
LORA_BYPASS_MARKERS = [
    "lightx2v_I2V_14B_480p_cfg_step_distill_rank64",
    "WanAnimate_relight_lora_fp16",
]


def patch_workflow(wf_path: Path) -> None:
    log("patching workflow JSON (value-based matching, not raw text)")
    data = json.loads(wf_path.read_text())
    fixed_count = 0
    bypassed_count = 0

    for node in data.get("nodes", []):
        wv = node.get("widgets_values")
        if wv is None:
            continue
        items = wv if isinstance(wv, list) else list(wv.values()) if isinstance(wv, dict) else []

        should_bypass = False
        for item in items:
            if not isinstance(item, str):
                continue
            for marker in LORA_BYPASS_MARKERS:
                if marker in item:
                    should_bypass = True

        if should_bypass:
            node["mode"] = 4
            bypassed_count += 1
            continue

        if isinstance(wv, list):
            for i, item in enumerate(wv):
                if not isinstance(item, str):
                    continue
                for bad_substr, good_value in VALUE_FIXES.items():
                    if bad_substr in item:
                        wv[i] = good_value
                        fixed_count += 1
                if VAE_CORRECT in item and item != VAE_CORRECT:
                    wv[i] = VAE_CORRECT
                    fixed_count += 1

    wf_path.write_text(json.dumps(data))
    log(f"workflow patched: {fixed_count} filename(s) fixed, {bypassed_count} LoRA node(s) bypassed")


def download_models():
    from huggingface_hub import hf_hub_download
    for repo, filename, subfolder, label in MODEL_FILES:
        dest_dir = MODELS_DIR / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / Path(filename).name
        if dest_path.exists():
            log(f"{label} already present, skipping")
            continue
        log(f"START: download {label}")
        t0 = time.time()
        path = hf_hub_download(repo_id=repo, filename=filename, token=os.environ.get("HF_TOKEN") or None)
        shutil.copy(path, dest_path)
        log(f"DONE ({round(time.time()-t0,1)}s): {label}")


def main():
    hw = detect_hardware()
    log("HARDWARE REPORT: " + json.dumps(hw))
    if hw["gpu_count"] == 0:
        log("FATAL: no GPU detected.")
        sys.exit(1)

    run_step("clone ComfyUI",
              ["git", "clone", "--depth", "1",
               "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_DIR)])
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

    download_models()

    workflows_dir = COMFY_DIR / "user" / "default" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    wf_path = workflows_dir / "wanvideo_WanAnimate_example_01.json"
    run_step("download WanAnimate example workflow",
              ["curl", "-sL", "-o", str(wf_path),
               "https://raw.githubusercontent.com/kijai/ComfyUI-WanVideoWrapper/main/example_workflows/wanvideo_WanAnimate_example_01.json"])
    patch_workflow(wf_path)

    run_step("download cloudflared",
              ["wget", "-q", "-O", "/tmp/cloudflared",
               "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"])
    subprocess.run(["chmod", "+x", "/tmp/cloudflared"], check=True)

    log("starting ComfyUI")
    comfy_proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188", "--enable-cors-header"],
        cwd=str(COMFY_DIR))
    time.sleep(25)
    if comfy_proc.poll() is not None:
        log(f"FATAL: ComfyUI died, code {comfy_proc.returncode}")
        sys.exit(1)
    log(f"ComfyUI alive pid={comfy_proc.pid}")

    def notify(message: str):
        try:
            subprocess.run(["curl", "-s", "-d", message,
                             "https://ntfy.sh/wan22studio-deepfake807-notify"],
                            capture_output=True, timeout=15)
        except Exception as e:
            log(f"notify failed: {e}")

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
    stop_topic = "wan22studio-deepfake807-stopsignal"
    deadline = time.time() + 25 * 60
    stopped_by_signal = False
    while time.time() < deadline:
        time.sleep(15)
        try:
            import urllib.request as _ur
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
