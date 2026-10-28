#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 06：相关分析（对应 3-1 相关分析）
==================================================================
【原 Excel/Power BI 实现内涵】
    相关分析衡量两个连续变量线性关系的方向与强度（Pearson 相关系数），
    是判断“投入—产出”“成本—收益”等之间是否联动的基础方法。

【Python 迁移要点】
    1. 读「原始数据」里的 y、x 两列。
    2. 计算 Pearson 相关系数，并做显著性检验(p 值)。
    3. 散点图 + 拟合直线，直观呈现相关方向与强度。
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "correlation", "3-1相关分析-参考.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "correlation")


def main():
    section("1. 读取原始数据")
    df = pd.read_excel(SRC, sheet_name="原始数据")
    df = df.dropna()
    print(df.describe())

    section("2. Pearson 相关分析")
    r, p = stats.pearsonr(df["x"], df["y"])
    sr, sp = stats.spearmanr(df["x"], df["y"])
    print(f"Pearson  r = {r:.4f}，p = {p:.2e}")
    print(f"Spearman r = {sr:.4f}，p = {sp:.2e}")
    strength = "强" if abs(r) > 0.6 else ("中等" if abs(r) > 0.3 else "弱")
    direction = "正" if r > 0 else "负"
    print(f"结论：x 与 y 呈{strength}{direction}相关。")

    section("3. 散点图 + 拟合直线")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["x"], df["y"], alpha=0.6, color="#4C72B0")
    b1, b0 = np.polyfit(df["x"], df["y"], 1)
    xs = np.linspace(df["x"].min(), df["x"].max(), 100)
    ax.plot(xs, b0 + b1 * xs, color="#d62728", lw=2,
            label=f"y={b0:.2f}+{b1:.3f}x")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"相关分析  (Pearson r={r:.3f})")
    ax.legend()
    p = save_fig(fig, "correlation.png", OUT)
    print("已保存：", p)


if __name__ == "__main__":
    main()
