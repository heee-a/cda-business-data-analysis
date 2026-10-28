#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 10：分布分析（对应 2、引申出的分析方法 → 4 分布分析）
==================================================================
【原 Excel/Power BI 实现内涵】分布分析看某个指标在不同取值上的频数/概率分布，
并配合四分位数、IQR、异常值上限等描述统计。教材把“门店客流量随时间分布”
和“客单价分布(含箱线统计)”放在一张表里。
【Python 迁移要点】用 openpyxl 精确取出两块区域：①门店客流量时序；
②客单价样本，做直方图+箱线图，并输出与教材一致的五数概括。
"""
import os
import sys
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "distribution", "4分布分析.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "distribution")


def main():
    section("1. 读取两块区域")
    wb = openpyxl.load_workbook(SRC, data_only=True)
    ws = wb.active
    traffic, price = [], []
    for row in ws.iter_rows(values_only=True):
        # 门店客流量 在 C 列(索引2)，数据从 row3 起
        if isinstance(row[2], (int, float)):
            traffic.append(row[2])
        # 客单价 在 U 列(索引20)，数值行
        if isinstance(row[20], (int, float)):
            price.append(row[20])
    traffic = pd.Series(traffic)
    price = pd.Series(price)
    print(f"门店客流量样本：{len(traffic)} 个；客单价样本：{len(price)} 个")
    q1, med, q3 = np.percentile(price, [25, 50, 75])
    iqr = q3 - q1
    print(f"客单价 五数概括：min={price.min()}, Q1={q1:.2f}, 中位={med:.2f}, "
          f"Q3={q3:.2f}, max={price.max()}, IQR={iqr:.2f}, 上限={q3+1.5*iqr:.2f}")

    section("2. 分布可视化")
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(range(len(traffic)), traffic.values, color="#4C72B0")
    ax[0, 0].set_title("门店客流量随时间分布"); ax[0, 0].set_xlabel("时点序号")
    ax[0, 1].hist(price, bins=20, color="#55a868", edgecolor="white")
    ax[0, 1].set_title("客单价直方图"); ax[0, 1].set_xlabel("客单价")
    ax[1, 0].boxplot(price.values, orientation="horizontal")
    ax[1, 0].set_title("客单价箱线图"); ax[1, 0].set_xlabel("客单价")
    ax[1, 1].axis("off")
    ax[1, 1].text(0.05, 0.8, "客单价描述统计", fontsize=12, fontweight="bold")
    stats_txt = (f"样本量: {len(price)}\n均值: {price.mean():.2f}\n"
                 f"标准差: {price.std():.2f}\n中位数: {med:.2f}\n"
                 f"Q1: {q1:.2f}  Q3: {q3:.2f}\nIQR: {iqr:.2f}\n"
                 f"异常上限: {q3+1.5*iqr:.2f}")
    ax[1, 1].text(0.05, 0.5, stats_txt, fontsize=10, va="top")
    p = save_fig(fig, "distribution.png", OUT)
    print("已保存：", p)


if __name__ == "__main__":
    main()
