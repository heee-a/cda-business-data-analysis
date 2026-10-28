#!/usr/bin/env python3
# Convert a Jupyter .ipynb notebook into a clean, runnable .py file.
# - Code cells are emitted as-is (with common notebook magics commented out).
# - Markdown cells are preserved as # comments so the "code essence / 代码内涵"
#   (the author's explanations) survives the conversion.
# - A header documents the origin notebook and how to run it.
import json, re, sys, os

MAGICS = (
    "%matplotlib", "%run", "%load", "%time", "%timeit", "%reset", "%pwd",
    "%cd", "%config", "%store", "%who", "%whos", "%pip", "!python", "!",
)

def comment_markdown(text):
    out = []
    for line in text.split("\n"):
        if line.strip() == "":
            out.append("#")
        else:
            out.append("# " + line)
    return "\n".join(out)

def clean_code(code):
    lines = code.split("\n")
    cleaned = []
    for ln in lines:
        stripped = ln.lstrip()
        # comment out IPython magics / shell escapes
        if any(stripped.startswith(m) for m in MAGICS):
            cleaned.append("# " + ln)
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)

def convert(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    parts = []
    # Header
    title = os.path.basename(nb_path)
    parts.append(f'"""')
    parts.append(f"Converted from Jupyter notebook: {title}")
    parts.append(f"Source: {os.path.relpath(nb_path)}")
    parts.append(f"")
    parts.append(f"本文件由原教材 Jupyter 笔记本转换而来，保留原作者的讲解（以 # 注释形式）。")
    parts.append(f"运行前请先安装依赖： pandas numpy scipy scikit-learn matplotlib statsmodels seaborn mlxtend 等。")
    parts.append(f'"""')
    parts.append("")
    parts.append("# ---- 标准库 / 第三方库导入约定 ----")
    parts.append("")
    for i, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))
        if ctype == "markdown":
            if src.strip():
                parts.append("")
                parts.append("# " + "=" * 60)
                parts.append(comment_markdown(src))
                parts.append("# " + "=" * 60)
        elif ctype == "code":
            parts.append("")
            parts.append("# --- code cell ---")
            parts.append(clean_code(src))
    return "\n".join(parts) + "\n"

def main():
    if len(sys.argv) < 3:
        print("usage: nb_to_py.py <in.ipynb> <out.py>")
        sys.exit(1)
    out = convert(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {sys.argv[2]} ({len(out)} bytes)")

if __name__ == "__main__":
    main()
