#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 09：构成分析（对应 2、引申出的分析方法 → 3 构成分析）
==================================================================
【原 Excel/Power BI 实现内涵】构成分析看“部分-整体”关系，如各产品销量占比。
Excel 用饼图/百分比堆积；Power BI 用饼图/环形图/堆积柱形。
【Python 迁移要点】用 openpyxl 定位“产品类型/销售量”表，画饼图 + 百分比条形图。
"""
import os
import sys
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "composition", "3构成分析.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "composition")


def main():
    section("1. 定位并读取“产品类型/销售量”表")
    wb = openpyxl.load_workbook(SRC, data_only=True)
    ws = wb.active
    cat_col = val_col = header_row = None
    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        for j, v in enumerate(row):
            if isinstance(v, str) and v.strip() == "产品类型":
                cat_col, val_col, header_row = j, j + 1, i
                break
        if header_row:
            break
    rows = []
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        if row[cat_col] is None:          # 空行即停止，避免读到下方其他表
            break
        if isinstance(row[cat_col], str) and isinstance(row[val_col], (int, float)):
            rows.append((row[cat_col], row[val_col]))
    df = pd.DataFrame(rows, columns=["产品类型", "销售量"])
    print(df)

    section("2. 构成：饼图 + 百分比堆积条形图")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].pie(df["销售量"], labels=df["产品类型"].astype(str), autopct="%1.1f%%",
              startangle=90)
    ax[0].set_title("各产品销售量占比（饼图）")
    pct = (df["销售量"] / df["销售量"].sum() * 100).round(1)
    ax[1].barh(df["产品类型"].astype(str)[::-1], pct[::-1], color="#55a868")
    ax[1].set_title("各产品销售量占比（%）"); ax[1].set_xlabel("占比(%)")
    p = save_fig(fig, "composition.png", OUT)
    print("已保存：", p)


if __name__ == "__main__":
    main()
