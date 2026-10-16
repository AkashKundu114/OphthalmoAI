import os
import re
import subprocess
from PIL import Image, ExifTags

print("--- Step 1: Scanning working tree for secrets, credentials, and local paths ---")

SECRET_PATTERNS = {
    "Gemini / Google API Key": re.compile(r"AIza[0-9A-Za-z\\-_]{35}|AQ\.[0-9A-Za-z\\-_]{40,}"),
    "Generic Private Key": re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
    "AWS Access Key": re.compile(r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}"),
    "Azure Client Secret / ID": re.compile(r"(?i)(?:azure[_-]?client[_-]?secret|azure[_-]?tenant[_-]?id|azure[_-]?subscription[_-]?id)\s*[:=]\s*['\"][0-9a-fA-F-]{20,}['\"]"),
    "Hardcoded JWT Secret": re.compile(r"(?i)jwt[_-]?secret(?:[_-]?key)?\s*=\s*['\"][0-9a-fA-F]{32,}['\"]"),
    "Hardcoded DB Password": re.compile(r"(?i)postgres(?:ql)?://[^:]+:([^@]+)@"),
    "Absolute Local Path Windows": re.compile(r"(?:[c-zC-Z]:\\(?:Users|OphthalmoAI|antigravity)[a-zA-Z0-9_\\.-]+)"),
}

EXCLUDE_DIRS = {".git", ".venv_gpu", "venv", "venv_gpu", "node_modules", ".pytest_cache", "__pycache__", "dist", "build", "REMEDIATION"}

working_tree_findings = []

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
    for f in files:
        if f.endswith((".pyc", ".png", ".jpg", ".jpeg", ".ico", ".wasm", ".zip", ".pdf", ".db", ".log")):
            continue
        filepath = os.path.join(root, f)
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()
                for idx, line in enumerate(lines, 1):
                    for name, pat in SECRET_PATTERNS.items():
                        m = pat.search(line)
                        if m:
                            working_tree_findings.append({
                                "file": filepath,
                                "line": idx,
                                "type": name,
                                "match": m.group(0)[:60]
                            })
        except Exception as e:
            pass

print(f"Working tree secret findings: {len(working_tree_findings)}")
for item in working_tree_findings[:30]:
    print(f"[{item['type']}] {item['file']}:{item['line']} -> {item['match']}")

print("\n--- Step 2: Scanning full git history for secrets ---")
try:
    cmd = ["git", "log", "-p", "--all"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore")
    history_findings = []
    current_commit = ""
    for line in proc.stdout:
        if line.startswith("commit "):
            current_commit = line.strip().split()[1]
        for name, pat in SECRET_PATTERNS.items():
            if name == "Absolute Local Path Windows":
                continue # ignore local paths in history diffs
            m = pat.search(line)
            if m:
                history_findings.append({
                    "commit": current_commit[:10],
                    "type": name,
                    "match": m.group(0)[:60]
                })
    proc.wait()
    print(f"History secret findings: {len(history_findings)}")
    unique_history = {}
    for h in history_findings:
        key = (h["type"], h["match"])
        if key not in unique_history:
            unique_history[key] = h["commit"]
    for (t, m), c in list(unique_history.items())[:20]:
        print(f"[{t}] Commit {c} -> {m}")
except Exception as e:
    print(f"Git history scan error: {e}")

print("\n--- Step 3: Scanning images for EXIF / GPS metadata ---")
exif_findings = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
            img_path = os.path.join(root, f)
            try:
                with Image.open(img_path) as im:
                    exif = im._getexif()
                    if exif:
                        gps_info = None
                        tags_present = []
                        for tag_id, value in exif.items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            tags_present.append(str(tag))
                            if tag == "GPSInfo":
                                gps_info = value
                        exif_findings.append({
                            "file": img_path,
                            "tags": tags_present[:5],
                            "gps": bool(gps_info)
                        })
            except Exception:
                pass

print(f"Images with EXIF data: {len(exif_findings)}")
for ef in exif_findings:
    print(f"EXIF found in {ef['file']}: tags={ef['tags']}, GPS={ef['gps']}")

if not exif_findings:
    print("Zero images with EXIF/GPS metadata detected.")
