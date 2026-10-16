import subprocess
import os
from collections import defaultdict

proc = subprocess.run(['git', 'ls-files'], capture_output=True, text=True)
files = proc.stdout.splitlines()

stats = defaultdict(lambda: {'files': 0, 'lines': 0, 'bytes': 0})
ext_map = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.jsx': 'JavaScript (React)',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript (React)',
    '.html': 'HTML',
    '.css': 'CSS',
    '.json': 'JSON',
    '.md': 'Markdown',
    '.yml': 'YAML',
    '.yaml': 'YAML',
    '.sh': 'Shell',
    '.ps1': 'PowerShell',
    '.sql': 'SQL',
    '.ini': 'INI',
    '.toml': 'TOML',
    '.txt': 'Text',
    '.dockerfile': 'Dockerfile'
}

for f in files:
    if not os.path.exists(f): continue
    name = os.path.basename(f)
    ext = os.path.splitext(f)[1].lower()
    lang = ext_map.get(ext, 'Other')
    if 'dockerfile' in name.lower(): lang = 'Dockerfile'
    
    stats[lang]['files'] += 1
    size = os.path.getsize(f)
    stats[lang]['bytes'] += size
    try:
        with open(f, 'rb') as fp:
            stats[lang]['lines'] += sum(1 for _ in fp)
    except:
        pass

print(f"{'Language':22} | {'Files':7} | {'Lines':10} | {'Bytes':12}")
print("-" * 60)
for lang, s in sorted(stats.items(), key=lambda x: x[1]['lines'], reverse=True):
    print(f"{lang:22} | {s['files']:7} | {s['lines']:10} | {s['bytes']:12,}")

total_files = sum(s['files'] for s in stats.values())
total_lines = sum(s['lines'] for s in stats.values())
total_bytes = sum(s['bytes'] for s in stats.values())
print("-" * 60)
print(f"{'TOTAL':22} | {total_files:7} | {total_lines:10} | {total_bytes:12,}")
