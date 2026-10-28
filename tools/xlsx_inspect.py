#!/usr/bin/env python3
# Minimal xlsx inspector using only the standard library (zipfile + xml).
# Lists sheets, dimensions, headers, and dumps a few sample rows + formulas.
import zipfile, re, sys
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

def col_to_num(col):
    n = 0
    for c in col:
        n = n * 26 + (ord(c) - ord('A') + 1)
    return n

def parse_ref(ref):
    m = re.match(r"([A-Z]+)(\d+)", ref)
    return m.group(1), int(m.group(2))

def load(path):
    z = zipfile.ZipFile(path)
    # shared strings
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall(f"{NS}si"):
            txt = "".join(t.text or "" for t in si.iter(f"{NS}t"))
            shared.append(txt)
    # workbook sheet names + rels
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    sheets = []
    for s in wb.findall(f"{NS}sheets/{NS}sheet"):
        sheets.append((s.get("name"), s.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")))
    # rels
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {}
    for r in rels:
        rid_to_target[r.get("Id")] = r.get("Target")
    result = []
    for name, rid in sheets:
        target = rid_to_target.get(rid)
        if not target.startswith("xl/"):
            target = "xl/" + target
        data = z.read(target)
        result.append((name, data))
    return shared, result

def sheet_to_rows(data, shared):
    root = ET.fromstring(data)
    rows = {}
    maxc = 0
    for c in root.iter(f"{NS}c"):
        ref = c.get("r")
        col, rownum = parse_ref(ref)
        t = c.get("t")
        v = c.find(f"{NS}v")
        f = c.find(f"{NS}f")
        val = None
        if t == "s" and v is not None:
            val = shared[int(v.text)]
        elif v is not None:
            val = v.text
        elif f is not None:
            val = f"=({f.text})"
        rows.setdefault(rownum, {})[col] = val
        maxc = max(maxc, col_to_num(col))
    return rows, maxc

def main():
    path = sys.argv[1]
    shared, sheets = load(path)
    for name, data in sheets:
        rows, maxc = sheet_to_rows(data, shared)
        print(f"\n===== SHEET: {name}  (rows={len(rows)}, maxcol={maxc}) =====")
        # header row = row 1
        keys = sorted(rows.get(1, {}).keys(), key=col_to_num)
        hdr = [rows[1].get(k, "") for k in keys]
        print("HEADER:", hdr)
        for r in sorted(rows.keys()):
            if r == 1: continue
            if r > 6: 
                print("   ... (%d more rows)" % (len(rows)-6)); break
            vals = [rows[r].get(k, "") for k in keys]
            print(f"  r{r}:", vals)

if __name__ == "__main__":
    main()
