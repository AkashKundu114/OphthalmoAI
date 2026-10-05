#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "         OPHTHALMOAI - POINT-OF-CARE RETINAL SCREENING"
echo "======================================================================"

PYTHON_EXE=""
if [ -f "$ROOT_DIR/.venv_gpu/bin/python" ]; then
    PYTHON_EXE="$ROOT_DIR/.venv_gpu/bin/python"
    echo "[*] Selected GPU Runtime: $ROOT_DIR/.venv_gpu"
elif [ -f "$ROOT_DIR/venv_gpu/bin/python" ]; then
    PYTHON_EXE="$ROOT_DIR/venv_gpu/bin/python"
    echo "[*] Selected GPU Runtime: $ROOT_DIR/venv_gpu"
elif [ -f "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$ROOT_DIR/.venv/bin/python"
    echo "[*] Selected CPU Runtime: $ROOT_DIR/.venv"
elif [ -f "$ROOT_DIR/venv/bin/python" ]; then
    PYTHON_EXE="$ROOT_DIR/venv/bin/python"
    echo "[*] Selected CPU Runtime: $ROOT_DIR/venv"
elif [ -f "$ROOT_DIR/.venv_gpu/Scripts/python.exe" ]; then
    PYTHON_EXE="$ROOT_DIR/.venv_gpu/Scripts/python.exe"
    echo "[*] Selected Windows GPU Runtime: $ROOT_DIR/.venv_gpu"
elif [ -f "$ROOT_DIR/venv_gpu/Scripts/python.exe" ]; then
    PYTHON_EXE="$ROOT_DIR/venv_gpu/Scripts/python.exe"
    echo "[*] Selected Windows GPU Runtime: $ROOT_DIR/venv_gpu"
elif [ -f "$ROOT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON_EXE="$ROOT_DIR/.venv/Scripts/python.exe"
    echo "[*] Selected Windows CPU Runtime: $ROOT_DIR/.venv"
elif [ -f "$ROOT_DIR/venv/Scripts/python.exe" ]; then
    PYTHON_EXE="$ROOT_DIR/venv/Scripts/python.exe"
    echo "[*] Selected Windows CPU Runtime: $ROOT_DIR/venv"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python3)"
    echo "[*] Selected System python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python)"
    echo "[*] Selected System python"
else
    echo "[!] Error: No Python executable found. Please create a virtual environment first."
    exit 1
fi

exec "$PYTHON_EXE" -u "$ROOT_DIR/scripts/start_app.py" "$@"
