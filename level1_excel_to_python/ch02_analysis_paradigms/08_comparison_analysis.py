#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 08：对比分析（对应 2、由基础分析范式引申出的分析方法 → 2 对比分析）
==================================================================
【原 Excel/Power BI 实现内涵】把不同类别(如各产品)的同一指标放在一起比较，
找出优势/劣势项。Excel 用条形图；Power BI 用条形/柱形视觉对象。
【Python 迁移要点】用 openpyxl 在含多张表的模板里定位“产品类型/订单量”表，
读取出后画排序条形图并标注数值。
"""
import os
import sys
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "comparison", "2对比分析.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "comparison")


def main():
    section("1. 定位并读取“产品类型/订单量”表")
    wb = openpyxl.load_workbook(SRC, data_only=True)
    ws = wb.active
    # 找到表头行：含“产品类型”的那一行，产品类型所在列即类别列，下一列是订单量
    cat_col = val_col = None
    header_row = None
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
    df = pd.DataFrame(rows, columns=["产品类型", "订单量"]).sort_values(
        "订单量", ascending=False).reset_index(drop=True)
    print(df)

    section("2. 对比条形图")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["产品类型"].astype(str), df["订单量"], color="#4C72B0")
    ax.set_title("不同产品订单量对比")
    ax.set_ylabel("订单量（个）")
    for i, v in enumerate(df["订单量"]):
        ax.text(i, v + max(df["订单量"]) * 0.01, str(int(v)), ha="center", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    p = save_fig(fig, "comparison.png", OUT)
    print("已保存：", p)
    print(f"最高：{df.iloc[0]['产品类型']}={int(df.iloc[0]['订单量'])}；"
          f"最低：{df.iloc[-1]['产品类型']}={int(df.iloc[-1]['订单量'])}；"
          f"倍差 {df.iloc[0]['订单量']/df.iloc[-1]['订单量']:.2f}x")


if __name__ == "__main__":
    main()
