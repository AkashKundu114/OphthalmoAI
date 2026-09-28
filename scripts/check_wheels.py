import urllib.request
import re

url = "https://download.pytorch.org/whl/torchvision/"
req = urllib.request.Request(url, headers={'User-Agent': 'pip/24.0'})
html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
links = [l for l in re.findall(r'href=[\'"]([^\'"]+)', html) if '0.24.1' in l and 'cu' in l and 'cp312' in l and 'linux' in l]
print(f"Found {len(links)} CUDA wheels for 0.24.1:")
for l in links:
    print(" ", l)
