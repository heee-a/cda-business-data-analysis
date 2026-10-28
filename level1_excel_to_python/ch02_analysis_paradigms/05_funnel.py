#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 05：漏斗图（对应 2-1 漏斗图）
==================================================================
【原 Excel/Power BI 实现内涵】
    漏斗分析用于刻画用户从“认知→接触→意向→明确→谈判→成交”的逐级转化，
    每一层相对上一层的转化率、以及相对首层的总体转化率，是运营诊断的利器。

【Python 迁移要点】
    1. 读 Sheet1 的阶段、用户数、转化率、总体转化率。
    2. 画漏斗（每层宽度随用户数递减）并标注转化率。
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "funnel", "2-1漏斗图-参考.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "funnel")


def main():
    section("1. 读取漏斗各阶段")
    df = pd.read_excel(SRC, sheet_name="Sheet1", header=2)
    df = df.dropna(subset=["阶段"]).reset_index(drop=True)
    df = df[["阶段", "用户数", "转化率", "总体转化率"]].copy()
    print(df)

    section("2. 计算总体转化率(以首层为100%)")
    top = df["用户数"].iloc[0]
    df["总体转化率%"] = (df["用户数"] / top * 100).round(1)
    print(df)

    section("3. 绘制漏斗图")
    fig, ax = plt.subplots(figsize=(8, 6))
    n = len(df)
    y = range(n)
    maxw = df["用户数"].max()
    for i, (_, row) in enumerate(df.iterrows()):
        w = row["用户数"] / maxw
        # 对称梯形表示漏斗层宽
        ax.barh(i, w, height=0.7, color=plt.cm.Blues(0.4 + 0.5 * i / n),
                edgecolor="white")
        label = f"{row['阶段']}  用户数={int(row['用户数'])}"
        if i > 0:
            label += f"  环比={row['转化率']:.2f}"
        label += f"  总体={row['总体转化率%']:.1f}%"
        ax.text(0.01, i, label, va="center", ha="left", fontsize=10, color="black")
    ax.set_yticks(list(y))
    ax.set_yticklabels([])
    ax.set_xlim(0, 1.05)
    ax.set_xticks([])
    ax.set_title("销售漏斗：各阶段用户数与转化率")
    p = save_fig(fig, "funnel.png", OUT)
    print("已保存：", p)

    out_csv = os.path.join(OUT, "funnel.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("已保存：", out_csv)


if __name__ == "__main__":
    main()
