#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3章 商业数据分析框架 —— 案例2：财务费用分析（Python 迁移）
=============================================================
原教材（Excel）对“期间费用”做框架式分析：
- 总费用随时间的趋势；
- 三大期间费用（管理费用 / 销售费用 / 财务费用）构成；
- 管理费用内部明细构成。

本脚本用 pandas + matplotlib 复刻上述框架分析。
注意：原始表中存在合并单元格产生的空列/表头错位，脚本按“列名定位”稳健读取。

运行：python 02_expense_analysis.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import section, save_fig, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "ch03_framework", "财务费用分析.xlsx")
OUT = os.path.join(os.path.dirname(__file__), "output")


def _clean_cols(df):
    """去掉全空列、规整列名。"""
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main():
    section("1. 读取三张费用明细表")
    total = _clean_cols(pd.read_excel(SRC, sheet_name="总费用数据"))
    three = _clean_cols(pd.read_excel(SRC, sheet_name="财务三大期间费用数据"))
    mgmt = _clean_cols(pd.read_excel(SRC, sheet_name="管理费用数据"))
    print("总费用表  :", total.shape, list(total.columns))
    print("三大费用表:", three.shape, list(three.columns))
    print("管理费用表:", mgmt.shape, list(mgmt.columns))

    section("2. 趋势分析：总费用随时间变化")
    total = total.dropna(subset=["年份", "金额"])
    total["金额"] = pd.to_numeric(total["金额"], errors="coerce")
    total["标签"] = total["年份"].astype(str) + "-" + total["月份"].astype(str).str.zfill(2)
    print(total[["标签", "金额"]].to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(total["标签"], total["金额"], marker="o", color="#4C72B0")
    ax.set_title("总费用趋势")
    ax.set_ylabel("金额")
    plt.xticks(rotation=45, ha="right")
    p1 = save_fig(fig, "expense_trend.png", OUT)
    print("已保存:", p1)

    section("3. 构成分析：三大期间费用占比")
    # 三大费用表的“期间费用项”列区分 管理费用/销售费用/财务费用
    three = three.dropna(subset=["期间费用项", "金额"])
    three["金额"] = pd.to_numeric(three["金额"], errors="coerce")
    by_type = three.groupby("期间费用项")["金额"].sum()
    # 仅保留三项主费用
    by_type = by_type[by_type.index.isin(["管理费用", "销售费用", "财务费用"])]
    print(by_type.round(2).to_string())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(by_type.values, labels=by_type.index, autopct="%1.1f%%",
           startangle=90, colors=plt.cm.Set2.colors)
    ax.set_title("三大期间费用构成")
    p2 = save_fig(fig, "expense_composition.png", OUT)
    print("已保存:", p2)

    section("4. 构成分析：管理费用内部明细")
    mgmt = mgmt.dropna(subset=["管理费用项", "金额"])
    mgmt["金额"] = pd.to_numeric(mgmt["金额"], errors="coerce")
    by_mgmt = mgmt.groupby("管理费用项")["金额"].sum().sort_values(ascending=False)
    print(by_mgmt.round(2).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(by_mgmt.index, by_mgmt.values, color="#55A868")
    ax.set_title("管理费用明细构成")
    ax.set_xlabel("金额")
    p3 = save_fig(fig, "expense_mgmt_detail.png", OUT)
    print("已保存:", p3)

    section("结论")
    print("财务费用分析框架 = 趋势（看走向） + 构成（看结构/占比），")
    print("与案例1（收入）共用同一套“框架式分析”方法论。")


if __name__ == "__main__":
    main()
