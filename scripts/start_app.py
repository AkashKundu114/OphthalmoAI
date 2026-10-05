"""
start_app.py - OphthalmoAI Universal Application Launcher

Orchestrates the FastAPI backend and Vite React frontend with:
- Automatic GPU (CUDA RTX 5060) vs CPU environment detection
- Node.js & npm verification
- Backend health-polling readiness verification (/health)
- Interactive ASCII status dashboard
- Automatic browser launch (configurable)
- Real-time multiplexed stdout logging
- Robust process tree teardown on Windows and Unix
- Optional --public mode connecting Cloudflare Tunnel, Hugging Face, & Vercel
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
LOGS_DIR = ROOT_DIR / "logs"

VENV_GPU_CANDIDATES = [
    ROOT_DIR / ".venv_gpu" / "Scripts" / "python.exe",
    ROOT_DIR / "venv_gpu" / "Scripts" / "python.exe",
    ROOT_DIR / ".venv_gpu" / "bin" / "python",
    ROOT_DIR / "venv_gpu" / "bin" / "python",
]
VENV_CPU_CANDIDATES = [
    ROOT_DIR / ".venv" / "Scripts" / "python.exe",
    ROOT_DIR / "venv" / "Scripts" / "python.exe",
    ROOT_DIR / ".venv" / "bin" / "python",
    ROOT_DIR / "venv" / "bin" / "python",
]


def find_python(force_cpu: bool = False) -> Path:
    """Finds the best available python interpreter for the project."""
    if not force_cpu:
        for candidate in VENV_GPU_CANDIDATES:
            if candidate.exists():
                return candidate

    for candidate in VENV_CPU_CANDIDATES:
        if candidate.exists():
            return candidate

    return Path(sys.executable)


def find_npm() -> str:
    """Detects npm command on Windows (npm.cmd) or Unix (npm)."""
    import shutil
    if sys.platform == "win32":
        cmd = shutil.which("npm.cmd") or shutil.which("npm")
        return cmd or "npm.cmd"
    cmd = shutil.which("npm")
    return cmd or "npm"


def kill_proc_tree(proc: subprocess.Popen | None) -> None:
    """Cleanly terminates a process and all of its spawned children."""
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


def pipe_stream(pipe, prefix: str, log_file=None):
    """Streams process output line by line with clear prefixes and disk logging."""
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            stripped = line.rstrip()
            print(f"{prefix} {stripped}", flush=True)
            if log_file:
                try:
                    log_file.write(line)
                    log_file.flush()
                except Exception:
                    pass
    except Exception:
        pass


def wait_for_backend(host: str, port: int, timeout_sec: int = 35) -> tuple[bool, str]:
    """Polls backend healthcheck until ready or timed out."""
    url = f"http://{host}:{port}/health"
    start = time.time()
    device = "Unknown"

    while time.time() - start < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OphthalmoAI-Launcher/1.0"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    import json
                    data = json.loads(resp.read().decode("utf-8"))
                    device = data.get("device", "CPU")
                    return True, device
        except Exception:
            time.sleep(0.6)

    return False, device


def print_banner(host: str, backend_port: int, frontend_port: int, device: str, is_public: bool = False, public_url: str = ""):
    banner = f"""
======================================================================
         OPHTHALMOAI - CLINICAL DECISION-SUPPORT PLATFORM
======================================================================
  Compute Engine:     {device.upper()}
  Backend API:        http://{host}:{backend_port}
  Swagger Docs:       http://{host}:{backend_port}/docs
  Frontend Web UI:    http://localhost:{frontend_port}
"""
    if is_public and public_url:
        banner += f"  Cloudflare Tunnel:  {public_url}\n"
        banner += "  Vercel Live App:    https://ophthalmo-ai-mu.vercel.app/\n"
        banner += "  Hugging Face Space: https://akashkundu114-ophthalmoai-demo.static.hf.space\n"

    banner += """======================================================================
  Press Ctrl+C to stop all services cleanly.
======================================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="OphthalmoAI Unified Service Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Backend host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--frontend-port", type=int, default=5176, help="Frontend port (default: 5176)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    parser.add_argument("--backend-only", action="store_true", help="Start only the FastAPI backend")
    parser.add_argument("--frontend-only", action="store_true", help="Start only the Vite frontend")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference mode")
    parser.add_argument("--public", action="store_true", help="Launch in public GPU mode (Cloudflare Tunnel + Vercel/HF sync)")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # If public GPU mode requested, delegate to start_public_gpu.py
    if args.public:
        script = ROOT_DIR / "scripts" / "start_public_gpu.py"
        python_exe = find_python(force_cpu=False)
        cmd = [str(python_exe), "-u", str(script)]
        sys.exit(subprocess.run(cmd, cwd=str(ROOT_DIR)).returncode)

    python_exe = find_python(force_cpu=args.cpu)
    npm_cmd = find_npm()

    print(f"[*] Root Directory:    {ROOT_DIR}")
    print(f"[*] Python Runtime:    {python_exe}")
    print(f"[*] Frontend Runtime:  {npm_cmd}")

    if not args.frontend_only and not python_exe.exists():
        print(f"[!] Python executable not found at {python_exe}. Run 'python -m venv venv' first.")
        sys.exit(1)

    if not args.backend_only and not (FRONTEND_DIR / "node_modules").exists():
        print("[*] Installing frontend dependencies (node_modules missing)...")
        subprocess.run([npm_cmd, "install"], cwd=str(FRONTEND_DIR), check=True)

    backend_proc: subprocess.Popen | None = None
    frontend_proc: subprocess.Popen | None = None
    backend_log = open(LOGS_DIR / "backend.log", "w", encoding="utf-8", errors="replace")
    frontend_log = open(LOGS_DIR / "frontend.log", "w", encoding="utf-8", errors="replace")

    def handle_shutdown(signum=None, frame=None):
        print("\n\n[*] Shutting down OphthalmoAI services gracefully...")
        if frontend_proc:
            kill_proc_tree(frontend_proc)
        if backend_proc:
            kill_proc_tree(backend_proc)
        try:
            backend_log.close()
            frontend_log.close()
        except Exception:
            pass
        print("[*] All processes stopped. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # 1. Start Backend
    backend_device = "CPU"
    if not args.frontend_only:
        print(f"\n[1/2] Starting FastAPI backend on http://{args.host}:{args.port}...")
        backend_env = os.environ.copy()
        backend_env["PYTHONUNBUFFERED"] = "1"
        if args.cpu:
            backend_env["FORCE_CPU"] = "true"

        backend_proc = subprocess.Popen(
            [
                str(python_exe),
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
            cwd=str(ROOT_DIR),
            env=backend_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        threading.Thread(
            target=pipe_stream,
            args=(backend_proc.stdout, "[BACKEND]", backend_log),
            daemon=True,
        ).start()

        # Wait for health
        print("  Waiting for backend initialization & model verification...")
        is_ready, device_name = wait_for_backend(args.host, args.port, timeout_sec=35)
        if not is_ready:
            print("[!] Warning: Backend health check timed out. Checking process status...")
            if backend_proc.poll() is not None:
                print(f"[!] Backend process exited with code {backend_proc.returncode}.")
                handle_shutdown()
        else:
            backend_device = device_name
            print(f"  [OK] Backend active on device: {backend_device}")

    # 2. Start Frontend
    final_frontend_url = f"http://localhost:{args.frontend_port}"
    if not args.backend_only:
        print(f"\n[2/2] Starting Vite React frontend on http://localhost:{args.frontend_port}...")
        frontend_env = os.environ.copy()
        frontend_env["VITE_DEV_BACKEND_URL"] = f"http://{args.host}:{args.port}"
        frontend_env["CI"] = "false"

        frontend_cmd = [
            npm_cmd,
            "run",
            "dev",
            "--",
            "--port",
            str(args.frontend_port),
            "--host",
            args.host,
        ]

        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=str(FRONTEND_DIR),
            env=frontend_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=(sys.platform == "win32"),
        )

        detected_url = [final_frontend_url]
        ready_event = threading.Event()
        url_pattern = re.compile(r"https?://(?:localhost|127\.0\.0\.1):\d+")

        def monitor_frontend(pipe):
            try:
                for line in iter(pipe.readline, ""):
                    if not line:
                        break
                    stripped = line.rstrip()
                    print(f"[FRONTEND] {stripped}", flush=True)
                    if frontend_log:
                        try:
                            frontend_log.write(line)
                            frontend_log.flush()
                        except Exception:
                            pass
                    if "Local:" in stripped or "Network:" in stripped:
                        match = url_pattern.search(stripped)
                        if match and not ready_event.is_set():
                            detected_url[0] = match.group(0)
                            ready_event.set()
            except Exception:
                pass

        threading.Thread(
            target=monitor_frontend,
            args=(frontend_proc.stdout,),
            daemon=True,
        ).start()

        # Wait up to 10 seconds for Vite to output its actual bound port
        ready_event.wait(timeout=10)
        final_frontend_url = detected_url[0]

    # Extract port from detected URL for banner
    port_match = re.search(r":(\d+)", final_frontend_url)
    display_frontend_port = int(port_match.group(1)) if port_match else args.frontend_port

    # Print dashboard banner
    print_banner(
        host=args.host,
        backend_port=args.port,
        frontend_port=display_frontend_port,
        device=backend_device,
    )

    # Open browser to the EXACT detected OphthalmoAI URL
    if not args.no_browser and not args.backend_only:
        print(f"[*] Opening {final_frontend_url} in your default browser...")
        try:
            webbrowser.open(final_frontend_url)
        except Exception:
            pass


    # Keep alive and monitor child processes
    try:
        while True:
            time.sleep(1)
            if backend_proc and backend_proc.poll() is not None:
                print("\n[!] Alert: Backend service terminated unexpectedly.")
                break
            if frontend_proc and frontend_proc.poll() is not None:
                print("\n[!] Alert: Frontend service terminated unexpectedly.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        handle_shutdown()


if __name__ == "__main__":
    main()
