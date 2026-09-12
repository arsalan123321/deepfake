"""
Wan2.2 Video Studio backend - GPU control + tunnel URL retrieval.
Throttles ntfy.sh calls to avoid 429 rate limiting.
"""
import subprocess
import time
import re
import urllib.request
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Wan2.2 Video Studio API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = REPO_ROOT / "kaggle" / "notebook"
KERNEL_REF = "deepfake807/wan22-worker"
NTFY_TOPIC = "wan22studio-deepfake807-notify"
NTFY_STOP_TOPIC = "wan22studio-deepfake807-stopsignal"

STATE = {"status": "OFFLINE", "tunnel_url": None, "last_checked": None}
NTFY_CACHE = {"last_poll_time": 0, "min_interval_s": 20}

class StatusResponse(BaseModel):
    status: str
    tunnel_url: Optional[str]
    last_checked: Optional[float]


def _run(cmd: list, timeout: int = 30):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def _check_ntfy_for_url() -> Optional[str]:
    now = time.time()
    if now - NTFY_CACHE["last_poll_time"] < NTFY_CACHE["min_interval_s"]:
        return None  # throttled -- skip this call, keep whatever we already have
    NTFY_CACHE["last_poll_time"] = now
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}/json?poll=1&since=10m")
        with urllib.request.urlopen(req, timeout=8) as resp:
            lines = resp.read().decode().strip().split("\n")
        latest_url = None
        for line in lines:
            if not line:
                continue
            msg = json.loads(line).get("message", "")
            m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", msg)
            if m:
                latest_url = m.group(0)
        return latest_url
    except Exception:
        return None


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/gpu/status", response_model=StatusResponse)
def gpu_status():
    code, out, err = _run(["kaggle", "kernels", "status", KERNEL_REF])
    STATE["last_checked"] = time.time()
    if code != 0:
        STATE["status"] = "ERROR"
    elif "RUNNING" in out:
        STATE["status"] = "RUNNING"
        url = _check_ntfy_for_url()
        if url:
            STATE["tunnel_url"] = url
    elif "COMPLETE" in out:
        STATE["status"] = "OFFLINE"
        STATE["tunnel_url"] = None
    elif "CANCEL" in out:
        STATE["status"] = "STOPPING"
    else:
        STATE["status"] = "OFFLINE"
        STATE["tunnel_url"] = None
    return StatusResponse(status=STATE["status"], tunnel_url=STATE["tunnel_url"],
                           last_checked=STATE["last_checked"])


@app.post("/api/gpu/start", response_model=StatusResponse)
def gpu_start():
    STATE["status"] = "STARTING"
    STATE["tunnel_url"] = None
    NTFY_CACHE["last_poll_time"] = 0  # allow immediate poll after a fresh start
    code, out, err = _run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR),
                            "--accelerator", "NvidiaTeslaT4"], timeout=60)
    if code != 0:
        STATE["status"] = "ERROR"
    return StatusResponse(status=STATE["status"], tunnel_url=STATE["tunnel_url"],
                           last_checked=time.time())


@app.post("/api/gpu/stop")
def gpu_stop():
    STATE["status"] = "STOPPING"
    debug_error = None
    debug_status_code = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{NTFY_STOP_TOPIC}",
                data=b"stop", method="POST",
                headers={"Content-Type": "text/plain"}
            )
            resp = urllib.request.urlopen(req, timeout=8)
            debug_status_code = resp.status
            debug_error = None
            break
        except Exception as e:
            debug_error = f"{type(e).__name__}: {e}"
            if "429" in str(e):
                time.sleep(3 * (attempt + 1))  # backoff and retry on rate limit
                continue
            break
    return {
        "status": STATE["status"],
        "tunnel_url": STATE["tunnel_url"],
        "last_checked": time.time(),
        "debug_ntfy_error": debug_error,
        "debug_ntfy_status_code": debug_status_code,
    }
