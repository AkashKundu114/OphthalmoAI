"""
start_public_gpu.py

1-Click launcher that:
1. Starts the OphthalmoAI FastAPI backend using CUDA on your NVIDIA RTX 5060 GPU.
2. Launches Cloudflare Tunnel to expose the backend over secure HTTPS.
3. Injects the active tunnel URL into frontend configuration (apiConfig.js and vercel.json).
4. Synchronizes and updates your live Hugging Face Space using the huggingface_hub SDK.
5. Deploys/syncs with Vercel hosting so the live web app points directly to your active GPU.
6. Keeps running and cleans up all processes on exit.
"""

from __future__ import annotations

import os
import re
import shutil
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


def shutil_which(cmd: str) -> str | None:
    return shutil.which(cmd)


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


def kill_proc_tree(proc: subprocess.Popen | None) -> None:
    """Terminates process and all children on Windows/Unix."""
    if proc is None or proc.poll() is not None:
        return
    pid = proc.pid
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return None


def sync_api_config(new_url: str) -> None:
    """Updates frontend/src/apiConfig.js with the active tunnel URL as fallback."""
    api_config_path = ROOT_DIR / "frontend" / "src" / "apiConfig.js"
    if not api_config_path.exists():
        return
    try:
        content = api_config_path.read_text(encoding="utf-8")
        updated = re.sub(
            r"export const FALLBACK_TUNNEL_URL = '.*?'",
            f"export const FALLBACK_TUNNEL_URL = '{new_url}'",
            content,
        )
        api_config_path.write_text(updated, encoding="utf-8")
        print(f"[SYNC] Updated frontend/src/apiConfig.js with fallback tunnel URL: {new_url}")
    except Exception as exc:
        print(f"[NOTE] apiConfig.js sync: {exc}")


def update_vercel_config(new_url: str) -> None:
    """Updates frontend/vercel.json rewrites for Vercel deployment."""
    frontend_dir = ROOT_DIR / "frontend"
    vercel_json_path = frontend_dir / "vercel.json"
    vercel_content = f'{{\n  "rewrites": [\n    {{\n      "source": "/api/:match*",\n      "destination": "{new_url}/:match*"\n    }},\n    {{\n      "source": "/(.*)",\n      "destination": "/index.html"\n    }}\n  ]\n}}\n'
    try:
        vercel_json_path.write_text(vercel_content, encoding="utf-8")
        print(f"[SYNC] Updated frontend/vercel.json with active tunnel destination: {new_url}")
    except Exception as exc:
        print(f"[NOTE] vercel.json sync: {exc}")


def deploy_to_vercel(new_url: str) -> None:
    """Syncs updated tunnel configuration with Vercel deployment."""
    print("\n[VERCEL] Syncing active GPU tunnel with Vercel...")
    frontend_dir = ROOT_DIR / "frontend"

    # 1. Try Vercel CLI if available
    vercel_cmd = shutil_which("vercel.cmd") or shutil_which("vercel")
    has_token = bool(os.environ.get("VERCEL_TOKEN"))
    if vercel_cmd or has_token:
        try:
            print("[VERCEL] Deploying directly via Vercel CLI...")
            npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
            res = subprocess.run(
                [npx_cmd, "--yes", "vercel", "deploy", "--prod", "--yes"],
                cwd=str(frontend_dir),
                capture_output=True,
                text=True,
                shell=(sys.platform == "win32"),
            )
            if res.returncode == 0:
                print(f"[VERCEL] Deployment completed successfully!\n{res.stdout[-200:].strip()}")
                return
            else:
                print(f"[VERCEL] CLI notice: {res.stderr[:200].strip()}")
        except Exception as err:
            print(f"[VERCEL] CLI notice: {err}")

    # 2. Sync via Git repository (triggers Vercel automatic CI/CD deployment)
    try:
        print("[VERCEL] Staging tunnel configuration in Git...")
        subprocess.run(
            ["git", "add", "frontend/vercel.json", "frontend/src/apiConfig.js", ".last_tunnel_url"],
            cwd=str(ROOT_DIR),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        diff_check = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(ROOT_DIR),
        )
        if diff_check.returncode != 0:
            commit_res = subprocess.run(
                ["git", "commit", "-m", f"chore(deploy): sync GPU tunnel {new_url} for Vercel & Hugging Face"],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
            )
            if commit_res.returncode == 0:
                print("[VERCEL] Committed updated tunnel routes.")
                push_res = subprocess.run(
                    ["git", "push", "origin", "main"],
                    cwd=str(ROOT_DIR),
                    capture_output=True,
                    text=True,
                )
                if push_res.returncode == 0:
                    print("[VERCEL] Pushed to origin/main -> Vercel is building the live deployment!")
                else:
                    print(f"[VERCEL] Git push notice: {push_res.stderr[:150].strip()}")
        else:
            print("[VERCEL] Git working tree already synchronized with active tunnel.")
    except Exception as exc:
        print(f"[VERCEL] Git deployment sync notice: {exc}")


def update_space_with_url(new_url: str) -> None:
    print(f"\n[SYNC] Updating Hugging Face Space & Vercel with GPU tunnel: {new_url}...")
    frontend_dir = ROOT_DIR / "frontend"

    # 1. Update source files
    sync_api_config(new_url)
    update_vercel_config(new_url)

    # 2. Rebuild frontend with the new VITE_API_URL
    print("[SYNC] Compiling frontend production bundle for Hugging Face...")
    build_env = os.environ.copy()
    build_env["VITE_API_URL"] = new_url
    npm_cmd = shutil_which("npm.cmd") or "npm.cmd"

    res = subprocess.run(
        [npm_cmd, "run", "build"],
        cwd=str(frontend_dir),
        env=build_env,
        capture_output=True,
        text=True,
        shell=(sys.platform == "win32"),
    )

    if res.returncode != 0:
        print(f"[WARNING] Frontend build issue: {res.stderr[:200]}")
    else:
        print("[SYNC] Frontend build successful.")

    # 3. Ensure Hugging Face metadata README.md is in dist
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

    # 4. Upload to Hugging Face Spaces
    hf_token = get_hf_token()
    uploaded = False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        print(f"[SYNC] Uploading to Hugging Face Space ({HF_SPACE}) via huggingface_hub...")
        api.upload_folder(
            folder_path=str(frontend_dir / "dist"),
            repo_id=HF_SPACE,
            repo_type="space",
            commit_message=f"Sync GPU tunnel: {new_url}",
        )
        print("[SYNC] Hugging Face Space is up to date!")
        uploaded = True
    except Exception as exc:
        print(f"[NOTE] huggingface_hub upload notice: {exc}")

    if not uploaded and HF_CLI.exists():
        try:
            print(f"[SYNC] Attempting upload via hf.exe CLI ({HF_SPACE})...")
            cli_res = subprocess.run(
                [str(HF_CLI), "upload", HF_SPACE, str(frontend_dir / "dist"), ".", "--repo-type", "space"],
                capture_output=True,
                text=True,
            )
            if cli_res.returncode == 0:
                print("[SYNC] Hugging Face Space updated via CLI!")
                uploaded = True
        except Exception as exc:
            print(f"[NOTE] HF CLI notice: {exc}")

    if not uploaded:
        print(f"[INFO] Hugging Face Space bundle prepared at {frontend_dir / 'dist'}.")
        print("       To enable auto-upload, set your HF_TOKEN in .env or run 'pip install huggingface_hub'.")

    # 5. Deploy / sync to Vercel
    deploy_to_vercel(new_url)

    save_last_url(new_url)


def is_backend_alive(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Checks if the FastAPI backend is already running and responsive."""
    import urllib.request
    try:
        req = urllib.request.Request(f"http://{host}:{port}/health", headers={"User-Agent": "OphthalmoAI/1.0"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def main():
    global CLOUDFLARED
    print("=" * 68)
    print("   OPHTHALMOAI - NVIDIA RTX 5060 GPU PUBLIC LAUNCHER & HOSTING SYNC")
    print("=" * 68)

    if not PYTHON_GPU.exists():
        print(f"[ERROR] GPU Python not found at {PYTHON_GPU}")
        sys.exit(1)

    if not CLOUDFLARED.exists():
        found = shutil_which("cloudflared") or shutil_which("cloudflared.exe")
        if found:
            CLOUDFLARED = Path(found)
        else:
            print(f"[ERROR] cloudflared not found at {CLOUDFLARED}")
            sys.exit(1)

    # 1. Start Backend Server if not already alive
    backend_proc = None
    if is_backend_alive():
        print("\n[1/3] PyTorch backend is already active & healthy on port 8000. Reusing instance.")
    else:
        print("\n[1/3] Launching PyTorch backend on RTX 5060 GPU (port 8000)...")
        backend_env = os.environ.copy()
        backend_env["PYTHONUNBUFFERED"] = "1"

        backend_proc = subprocess.Popen(
            [str(PYTHON_GPU), "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(ROOT_DIR),
            env=backend_env,
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
        if backend_proc is not None:
            kill_proc_tree(backend_proc)
        sys.exit(1)

    print(f"[2/3] Public GPU Tunnel Active: {tunnel_url}")

    # 3. Check if Space & Vercel need updating
    last_url = get_last_url()
    if tunnel_url != last_url:
        print(f"\n[3/3] Tunnel URL changed (Old: {last_url} -> New: {tunnel_url}).")
        update_space_with_url(tunnel_url)
    else:
        print("\n[3/3] Tunnel URL matches previous session. Deployments are already aligned.")

    print("\n" + "=" * 68)
    print("   ONLINE & READY FOR CLINICAL SCANS!")
    print("=" * 68)
    print("   * Primary Web App (Vercel):  https://ophthalmo-ai-mu.vercel.app/")
    print(f"   * Hugging Face Mirror:       https://akashkundu114-ophthalmoai-demo.static.hf.space")
    print(f"   * Hugging Face Space:        https://huggingface.co/spaces/{HF_SPACE}")
    print("   * Local Compute Engine:      NVIDIA GeForce RTX 5060 (CUDA Active)")
    print(f"   * Secure Cloudflare Tunnel:  {tunnel_url}")
    print("=" * 68)
    print("\nKeep this window open while you or visitors use the site.")
    print("Press Ctrl+C to stop the GPU server and tunnel.\n")

    def handle_exit(signum=None, frame=None):
        print("\nStopping services...")
        if backend_proc is not None:
            kill_proc_tree(backend_proc)
        if tunnel_proc is not None:
            kill_proc_tree(tunnel_proc)
        print("All services stopped cleanly. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        while True:
            time.sleep(1)
            if backend_proc and backend_proc.poll() is not None:
                print("\n[ALERT] Backend process terminated.")
                break
            if tunnel_proc and tunnel_proc.poll() is not None:
                print("\n[ALERT] Cloudflare tunnel terminated.")
                break
    except KeyboardInterrupt:
        handle_exit()


if __name__ == "__main__":
    main()
