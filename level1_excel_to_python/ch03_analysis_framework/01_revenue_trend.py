#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3章 商业数据分析框架 —— 案例1：收入趋势分析（Python 迁移）
=============================================================
原教材（Excel）演示“商业数据分析框架”：先用趋势分析看整体走向，
再用构成分析看省份结构，最后用对比分析看年际差异。

本脚本用 pandas + matplotlib 复刻：
1. 趋势分析：按 年-月 汇总销售额，看整体增长走势 + 3 期移动平均；
2. 构成分析：各销售省份销售额占比（饼图/条形）；
3. 对比分析：不同年份的销售额对比。

运行：python 01_revenue_trend.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import section, save_fig, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "ch03_framework", "收入趋势分析.xlsx")
OUT = os.path.join(os.path.dirname(__file__), "output")


def main():
    section("1. 读取数据")
    df = pd.read_excel(SRC, sheet_name="原始数据")
    df["销售额"] = pd.to_numeric(df["销售额"], errors="coerce")
    print("数据量:", df.shape, "列:", list(df.columns))
    print(df.head(3).to_string())

    section("2. 趋势分析：年-月 销售额走势 + 移动平均")
    df["年月"] = df["订单年份"].astype(str) + "-" + df["订单月份"].astype(str).str.zfill(2)
    monthly = df.groupby("年月")["销售额"].sum().reset_index()
    monthly = monthly.sort_values("年月")
    monthly["移动平均3期"] = monthly["销售额"].rolling(3, min_periods=1).mean()
    print(monthly.tail(6).to_string())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(monthly["年月"], monthly["销售额"], marker="o", label="月度销售额", color="#4C72B0")
    ax.plot(monthly["年月"], monthly["移动平均3期"], linestyle="--",
            label="3期移动平均", color="#C44E52")
    ax.set_title("收入趋势分析：月度销售额走势")
    ax.set_ylabel("销售额")
    ax.legend()
    plt.xticks(rotation=60, ha="right")
    p1 = save_fig(fig, "revenue_trend.png", OUT)
    print("已保存:", p1)

    section("3. 构成分析：各销售省份销售额占比")
    by_prov = df.groupby("销售省份")["销售额"].sum().sort_values(ascending=False)
    by_prov_pct = (by_prov / by_prov.sum() * 100).round(1)
    print(by_prov_pct.to_string())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_prov.index, by_prov.values / 1e4, color="#55A868")
    ax.set_title("各省份销售额构成（万元）")
    ax.set_ylabel("销售额(万元)")
    plt.xticks(rotation=45, ha="right")
    p2 = save_fig(fig, "revenue_by_province.png", OUT)
    print("已保存:", p2)

    section("4. 对比分析：不同年份销售额对比")
    by_year = df.groupby("订单年份")["销售额"].sum()
    print(by_year.to_string())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(by_year.index.astype(str), by_year.values / 1e4, color="#8172B3")
    ax.set_title("各年度销售额对比（万元）")
    ax.set_ylabel("销售额(万元)")
    p3 = save_fig(fig, "revenue_by_year.png", OUT)
    print("已保存:", p3)

    section("结论")
    print("趋势↑、构成（省份集中度）、对比（年际增长）三者结合，")
    print("即构成教材第3章“先趋势、再构成、后对比”的商业分析框架。")


if __name__ == "__main__":
    main()
