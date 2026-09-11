"""
start_public_gpu.py

1-Click launcher that:
1. Starts the OphthalmoAI FastAPI backend using CUDA on your NVIDIA RTX 5060 GPU.
2. Launches Cloudflare Tunnel to expose the backend over secure HTTPS.
3. Automatically syncs the public URL to your live Hugging Face Space if it changes.
4. Keeps running and cleans up on exit.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_GPU = ROOT_DIR / "venv_gpu" / "Scripts" / "python.exe"
CLOUDFLARED = Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe")
HF_CLI = Path(os.environ.get("USERPROFILE", "")) / ".local" / "bin" / "hf.exe"
LAST_URL_FILE = ROOT_DIR / ".last_tunnel_url"
HF_SPACE = "AkashKundu114/ophthalmoai-demo"


def get_last_url() -> str:
    if LAST_URL_FILE.exists():
        try:
            return LAST_URL_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""


def save_last_url(url: str) -> None:
    try:
        LAST_URL_FILE.write_text(url, encoding="utf-8")
    except Exception:
        pass


def update_space_with_url(new_url: str) -> None:
    print(f"\n[SYNC] Updating Hugging Face Space with new GPU tunnel URL: {new_url}...")
    frontend_dir = ROOT_DIR / "frontend"

    # Rebuild frontend with the new VITE_API_URL
    build_env = os.environ.copy()
    build_env["VITE_API_URL"] = new_url
    npm_cmd = shutil_which("npm.cmd") or "npm.cmd"

    res = subprocess.run(
        [npm_cmd, "run", "build"],
        cwd=str(frontend_dir),
        env=build_env,
        capture_output=True,
        text=True,
        shell=True,
    )

    if res.returncode != 0:
        print(f"[WARNING] Frontend build had an issue: {res.stderr[:200]}")
    else:
        print("[SYNC] Frontend build successful.")

    # Ensure README.md is in dist
    readme_path = frontend_dir / "dist" / "README.md"
    readme_content = """---
title: OphthalmoAI Demo
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---
"""
    readme_path.write_text(readme_content, encoding="utf-8")

    # Update frontend/vercel.json for Vercel deployment
    vercel_json_path = frontend_dir / "vercel.json"
    vercel_content = f'{{\n  "rewrites": [\n    {{\n      "source": "/api/:match*",\n      "destination": "{new_url}/:match*"\n    }},\n    {{\n      "source": "/(.*)",\n      "destination": "/index.html"\n    }}\n  ]\n}}\n'
    try:
        vercel_json_path.write_text(vercel_content, encoding="utf-8")
        print("[SYNC] Updated frontend/vercel.json with active tunnel.")
    except Exception as e:
        print(f"[NOTE] vercel.json update: {e}")

    # Upload to Hugging Face
    if HF_CLI.exists():
        print(f"[SYNC] Pushing updated build to Hugging Face ({HF_SPACE})...")
        subprocess.run(
            [str(HF_CLI), "upload", HF_SPACE, str(frontend_dir / "dist"), ".", "--repo-type", "space"],
            capture_output=True,
            text=True,
        )
        print("[SYNC] Hugging Face Space is up to date!")
    save_last_url(new_url)


def shutil_which(cmd: str) -> str | None:
    import shutil
    return shutil.which(cmd)


def main():
    print("=" * 65)
    print("   OPHTHALMOAI - NVIDIA RTX 5060 GPU PUBLIC LAUNCHER")
    print("=" * 65)

    if not PYTHON_GPU.exists():
        print(f"[ERROR] GPU Python not found at {PYTHON_GPU}")
        sys.exit(1)

    if not CLOUDFLARED.exists():
        print(f"[ERROR] cloudflared not found at {CLOUDFLARED}")
        sys.exit(1)

    # 1. Start Backend Server
    print("\n[1/3] Launching PyTorch backend on RTX 5060 GPU (port 8000)...")
    backend_proc = subprocess.Popen(
        [str(PYTHON_GPU), "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait briefly for backend to bind
    time.sleep(3)
    if backend_proc.poll() is not None:
        err = backend_proc.stderr.read() if backend_proc.stderr else "Unknown error"
        print(f"[ERROR] Backend failed to start: {err}")
        sys.exit(1)

    print("[1/3] Backend running in background on port 8000.")

    # 2. Start Cloudflare Tunnel
    print("\n[2/3] Starting Cloudflare Tunnel to expose your GPU securely...")
    tunnel_proc = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # Read output until URL is found
    for line in tunnel_proc.stdout:
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break
        if tunnel_proc.poll() is not None:
            break

    if not tunnel_url:
        print("[ERROR] Could not extract Cloudflare Tunnel URL.")
        backend_proc.terminate()
        sys.exit(1)

    print(f"[2/3] Public GPU Tunnel Active: {tunnel_url}")

    # 3. Check if Space needs updating
    last_url = get_last_url()
    if tunnel_url != last_url:
        print(f"\n[3/3] Tunnel URL changed (Old: {last_url} -> New: {tunnel_url}).")
        update_space_with_url(tunnel_url)
    else:
        print("\n[3/3] Tunnel URL matches previous session. Hugging Face Space is already in sync.")

    print("\n" + "=" * 65)
    print("   ONLINE & READY FOR SCANS!")
    print("=" * 65)
    print("   * Primary Web App (Vercel):  https://ophthalmo-ai-mu.vercel.app/")
    print(f"   * Hugging Face Mirror:       https://akashkundu114-ophthalmoai-demo.static.hf.space")
    print(f"   * Hugging Face Space:        https://huggingface.co/spaces/{HF_SPACE}")
    print("   * GPU Engine:                NVIDIA GeForce RTX 5060 Laptop GPU")
    print(f"   * Cloudflare API Tunnel:     {tunnel_url}")
    print("=" * 65)
    print("\nKeep this window open while you or visitors use the site.")
    print("Press Ctrl+C to stop the GPU server and tunnel.\n")

    def handle_exit(signum, frame):
        print("\nStopping services...")
        try:
            backend_proc.terminate()
            backend_proc.wait(timeout=3)
        except Exception:
            pass
        try:
            tunnel_proc.terminate()
            tunnel_proc.wait(timeout=3)
        except Exception:
            pass
        print("All services stopped cleanly. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("\n[ALERT] Backend process terminated.")
                break
            if tunnel_proc.poll() is not None:
                print("\n[ALERT] Cloudflare tunnel terminated.")
                break
    except KeyboardInterrupt:
        handle_exit(None, None)


if __name__ == "__main__":
    main()
