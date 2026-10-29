#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5章 业务数据分析 —— 银行案例：申请渠道归因分析（Python 迁移）
================================================================
原教材（WPS/Power BI）演示“归因分析”：把总体指标的变化拆解到
各个驱动因素（此处为“申请渠道”），回答“总量变化由谁贡献”。

本脚本用贷款申请表数据复刻归因分析：
1. 渠道构成：各申请渠道的占比（谁贡献最多）；
2. 时间趋势：各渠道随月份的申请量走势（看结构漂移）；
3. 因素拆解：以相邻时期对比，计算各渠道对“总量变化”的贡献度。

运行：python 01_attribution_analysis.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import section, save_fig, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "ch05_attribution", "归因分析数据.xlsx")
OUT = os.path.join(os.path.dirname(__file__), "output")


def main():
    section("1. 读取贷款申请表")
    df = pd.read_excel(SRC, sheet_name="贷款申请表")
    df["申请日期"] = pd.to_datetime(df["申请日期"], errors="coerce")
    # 解析年-月，便于时间序列归因
    df["年份"] = df["申请日期"].dt.year
    df["月份"] = df["申请日期"].dt.month
    df["年月"] = df["申请日期"].dt.to_period("M").astype(str)
    print("数据量:", df.shape, "列:", list(df.columns))
    print(df.head(3).to_string())

    section("2. 渠道构成：各申请渠道占比（谁贡献最多）")
    by_ch = df.groupby("申请渠道")["申请书编号（主键）"].count()
    by_ch_pct = (by_ch / by_ch.sum() * 100).round(1)
    print(by_ch_pct.to_string())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(by_ch_pct.values, labels=by_ch_pct.index, autopct="%1.1f%%",
           startangle=90, colors=plt.cm.Set2.colors)
    ax.set_title("贷款申请渠道构成")
    p1 = save_fig(fig, "attr_channel_share.png", OUT)
    print("已保存:", p1)

    section("3. 时间趋势：各渠道月度申请量（结构漂移）")
    pivot = pd.pivot_table(df, values="申请书编号（主键）", index="年月",
                           columns="申请渠道", aggfunc="count", fill_value=0)
    pivot = pivot.sort_index()
    print(pivot.tail(6).to_string())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for ch in pivot.columns:
        ax.plot(pivot.index, pivot[ch], marker="o", label=ch)
    ax.set_title("各申请渠道月度申请量趋势")
    ax.set_ylabel("申请量")
    ax.legend()
    plt.xticks(rotation=60, ha="right")
    p2 = save_fig(fig, "attr_channel_trend.png", OUT)
    print("已保存:", p2)

    section("4. 因素拆解：相邻半年对比，各渠道对总量变化的贡献")
    # 以前后两个时间段（如 上半年 vs 下半年，或首月 vs 末月）做增量归因
    period = df["申请日期"].dt.to_period("Q")  # 按季度切分
    grp = pd.pivot_table(df, values="申请书编号（主键）", index=period,
                         columns="申请渠道", aggfunc="count", fill_value=0)
    grp = grp.sort_index()
    if len(grp) >= 2:
        base, curr = grp.iloc[0], grp.iloc[-1]
        delta_total = (curr.sum() - base.sum())
        contrib = (curr - base).sort_values(ascending=False)
        contrib_pct = (contrib / delta_total * 100).round(1) if delta_total else contrib
        print(f"对比期: {grp.index[0]} -> {grp.index[-1]}")
        print(f"总量变化: {delta_total:+d}")
        print("各渠道贡献度(%):")
        print(contrib_pct.to_string())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#C44E52" if v > 0 else "#55A868" for v in contrib.values]
        ax.bar(contrib.index, contrib.values, color=colors)
        ax.set_title(f"渠道对总量变化的贡献（{grp.index[0]}→{grp.index[-1]}）")
        ax.set_ylabel("申请量变化")
        plt.xticks(rotation=20)
        p3 = save_fig(fig, "attr_contribution.png", OUT)
        print("已保存:", p3)

    section("结论")
    print("归因分析的核心：把“总量变化”按比例拆解到各驱动因素（渠道），")
    print("从而定位增长/下滑的主要来源——这是业务看板之外更进一步的诊断分析。")


if __name__ == "__main__":
    main()
