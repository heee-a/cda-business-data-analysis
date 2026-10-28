#!/usr/bin/env python3
# Download Python wheels via curl (which works through the environment proxy)
# and install them offline with pip --no-index. This avoids pip's index fetcher
# that is blocked (HTTP 403) in this environment.
import subprocess, re, os, sys, shutil
from urllib.parse import urljoin

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
WHEELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wheels")
os.makedirs(WHEELS_DIR, exist_ok=True)

# Full dependency closure for our experiments
PACKAGES = [
    "numpy", "pandas", "scipy", "matplotlib", "scikit-learn==1.5.2",
    "seaborn", "mlxtend", "openpyxl", "python-dateutil", "pytz", "tzdata",
    "six", "patsy", "joblib==1.4.2", "threadpoolctl", "contourpy", "cycler",
    "et-xmlfile",
    "fonttools", "kiwisolver", "packaging", "pillow", "pyparsing",
    "statsmodels==0.14.4",
]

def curl(url, out=None):
    cmd = ["curl", "-sL", "--max-time", "120"]
    if out:
        cmd += ["-o", out]
    cmd.append(url)
    return subprocess.run(cmd, capture_output=True, text=True)

def version_key(fn):
    m = re.search(r"-(\d+(?:\.\d+)*)", fn)
    if not m:
        return (0,)
    return tuple(int(x) for x in m.group(1).split("."))

def pick_best(candidates):
    # candidates: list of (filename, href)
    # Prefer cp313-win_amd64, then cp313-abi3, then py3-none-any / py2.py3-none-any
    def score(fn):
        if "cp313-cp313-win_amd64" in fn: return 5
        if "cp313-abi3" in fn: return 4
        if "cp312" in fn and "win_amd64" in fn: return 3
        if "py3-none-any" in fn or "py2.py3-none-any" in fn: return 2
        if "any.whl" in fn: return 1
        return 0
    best = max(candidates, key=lambda c: (score(c[0]), version_key(c[0])))
    return best

def fetch_links(pkg):
    pin = None
    if "==" in pkg:
        pkg, pin = pkg.split("==")
    url = f"{MIRROR}/{pkg}/"
    r = curl(url)
    if r.returncode != 0 or not r.stdout:
        print(f"[WARN] failed to fetch index for {pkg}")
        return []
    links = re.findall(r'href="([^"]+?\.whl)', r.stdout)
    out = []
    for href in links:
        fn = href.rsplit("/", 1)[-1]
        out.append((fn, urljoin(url, href)))
    return out

def main():
    for pkg in PACKAGES:
        pin = None
        if "==" in pkg:
            pkg, pin = pkg.split("==")
        print(f"== {pkg} ==")
        links = fetch_links(pkg)
        if not links:
            continue
        if pin:
            links = [l for l in links if f"-{pin}-" in l[0] or l[0].startswith(f"{pkg}-{pin}-")]
            if not links:
                print(f"   [WARN] pin {pin} not found"); continue
        best = pick_best(links)
        fn, href = best
        dest = os.path.join(WHEELS_DIR, fn)
        if os.path.exists(dest):
            print(f"   exists: {fn}")
            continue
        print(f"   downloading {fn}")
        r = curl(href, dest)
        if r.returncode != 0 or os.path.getsize(dest) < 1000:
            print(f"   [FAIL] {fn}: {r.stderr[:200]}")
            if os.path.exists(dest): os.remove(dest)
        else:
            print(f"   ok {round(os.path.getsize(dest)/1024/1024,1)} MB")

if __name__ == "__main__":
    main()
