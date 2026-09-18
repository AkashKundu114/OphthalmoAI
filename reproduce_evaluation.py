#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys

if __name__ == '__main__':
    s = Path(__file__).resolve().parent / 'scripts' / 'reproduce_evaluation.py'
    sys.exit(subprocess.call([sys.executable, str(s)] + sys.argv[1:]))
