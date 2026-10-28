#!/usr/bin/env python3
# Extract every .ipynb from the CDA textbook zip, convert it to a clean .py
# (preserving the author's markdown explanations as comments), organize the .py
# files by textbook/level, and pull in ONLY the data files that the notebooks
# actually reference (csv/xlsx), skipping files larger than MAX_DATA_MB.
#
# Output layout (inside cda-business-data-analysis/):
#   level2_quant_strategy_python/<chapter>/<notebook>.py
#   level3_agile_mining_python/<chapter>/<notebook>.py
#   level4_finance_cases_python/<chapter>/<notebook>.py
#   data/level2/...  data/level3/...  data/level4/...  (referenced, small files)
import json, os, re, sys, shutil
from urllib.parse import unquote

ZIP = r"D:/Users/HP/Desktop/商业数据分析/20260830 CDA教材资料2025.zip"
REPO = r"E:/workbuddy/2026-10-28-16-54-39/cda-business-data-analysis"
MAX_DATA_MB = 50.0

# map top-level textbook folder -> (target level dir, data dir)
TEXTBOOK_MAP = {
    "2《CDA 二级认证教材：量化策略分析》": ("level2_quant_strategy_python", "data/level2"),
    "3《CDA 三级认证教材：敏捷数据挖掘》": ("level3_agile_mining_python", "data/level3"),
    "4《金融数据分析和数据挖掘案例实战》": ("level4_finance_cases_python", "data/level4"),
}

MAGICS = ("%matplotlib","%run","%load","%time","%timeit","%reset","%pwd","%cd",
          "%config","%store","%who","%whos","%pip","!python","!")

def comment_md(text):
    out=[]
    for line in text.split("\n"):
        out.append("# " + line if line.strip() else "#")
    return "\n".join(out)

def clean_code(code):
    out=[]
    for ln in code.split("\n"):
        s=ln.lstrip()
        if any(s.startswith(m) for m in MAGICS):
            out.append("# " + ln); continue
        out.append(ln)
    return "\n".join(out)

def convert_nb(nb):
    cells=nb.get("cells",[])
    parts=[]
    parts.append('"""Converted from a CDA textbook Jupyter notebook.')
    parts.append("原作者的讲解以 # 注释形式保留，便于理解代码内涵。")
    parts.append('"""')
    parts.append("")
    for cell in cells:
        ct=cell.get("cell_type"); src="".join(cell.get("source",[]))
        if ct=="markdown":
            if src.strip():
                parts.append("\n# "+"="*60)
                parts.append(comment_md(src))
                parts.append("# "+"="*60)
        elif ct=="code":
            parts.append("\n# --- code cell ---")
            parts.append(clean_code(src))
    return "\n".join(parts)+"\n"

def find_referenced_data(code):
    # find pd.read_csv('x'), pd.read_excel('x'), open('x'), and path strings
    names=set()
    for m in re.findall(r"(?:read_csv|read_excel|read_table|to_csv|open)\s*\(\s*['\"]([^'\"]+)['\"]", code):
        names.add(m)
    return names

def main():
    import zipfile
    z=zipfile.ZipFile(ZIP)
    names=z.namelist()
    # index all data files by basename
    data_by_base={}
    for n in names:
        if n.endswith("/"): continue
        b=os.path.basename(n)
        # only data we care about
        if b.lower().endswith((".csv",".xlsx",".xls",".sav",".txt")):
            data_by_base.setdefault(b, []).append(n)
    # ipynb entries
    nb_entries=[n for n in names if n.endswith(".ipynb")]
    refs_total=0
    converted=0
    for nbpath in nb_entries:
        # determine textbook
        parts=nbpath.split("/")
        top=parts[1] if len(parts)>1 else ""
        if top not in TEXTBOOK_MAP:
            continue
        leveldir, datadir = TEXTBOOK_MAP[top]
        # chapter = the folder right before the notebook
        chapter = parts[-2] if len(parts)>=2 else "root"
        chapter_safe = re.sub(r'[\\/:*?"<>|]', "_", chapter)
        nbname = os.path.basename(nbpath)
        pyname = nbname[:-6]+".py"
        # read + convert
        data=z.read(nbpath)
        try:
            nb=json.loads(data)
        except Exception as e:
            print("  [SKIP json] %s : %s"%(nbpath,e)); continue
        pycode=convert_nb(nb)
        outdir=os.path.join(REPO, leveldir, chapter_safe)
        os.makedirs(outdir, exist_ok=True)
        outpath=os.path.join(outdir, pyname)
        with open(outpath,"w",encoding="utf-8") as f:
            f.write(pycode)
        converted+=1
        # referenced data
        code="".join("".join(c.get("source",[])) for c in nb.get("cells",[]) if c.get("cell_type")=="code")
        for ref in find_referenced_data(code):
            b=os.path.basename(ref)
            if b in data_by_base:
                for src in data_by_base[b]:
                    dest=os.path.join(REPO, datadir, b)
                    if os.path.exists(dest): continue
                    info=z.getinfo(src)
                    if info.file_size > MAX_DATA_MB*1024*1024:
                        print(f"  [SKIP large {round(info.file_size/1024/1024,1)}MB] {b}")
                        continue
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with z.open(src) as fi, open(dest,"wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    refs_total+=1
    print(f"Converted {converted} notebooks; extracted {refs_total} referenced data files.")
    # write README per level
    for top,(leveldir,datadir) in TEXTBOOK_MAP.items():
        ld=os.path.join(REPO, leveldir)
        if not os.path.isdir(ld): continue
        lines=[f"# {top} — Python 版（由 Jupyter 笔记本转换）\n"]
        for root,_,files in os.walk(ld):
            for f in sorted(files):
                if f.endswith(".py"):
                    rel=os.path.relpath(os.path.join(root,f), ld)
                    lines.append(f"- `{rel}`")
        with open(os.path.join(ld,"README.md"),"w",encoding="utf-8") as fh:
            fh.write("\n".join(lines)+"\n")
        print("wrote", os.path.join(leveldir,"README.md"))

if __name__=="__main__":
    main()
