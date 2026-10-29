#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 12（第10章）：效应分解法做预报（对应「使用效应分解方法做预报」）
==================================================================
【原 Excel/Power BI 实现内涵】
    效应分解法把时间序列拆成：趋势效应(Trend) + 季节效应(Seasonal) + 不规则
    效应(Irregular，残差)。教材用 Power BI / Excel 对两类数据做分解：
      · AirPassengers（1949-1960 月度国际航线客流）——经典乘法分解示例；
      · 信用卡消费额（含节日 / 去除节日 两份日度数据）——对比节日对序列的冲击。
    其内涵是：先估计趋势(移动平均)，再求季节指数，最后得到残差；预报时
    用“趋势×季节”重建。

【Python 迁移要点】
    1. 用 pandas 读入序列并构造时间索引。
    2. 用 statsmodels.seasonal_decompose 做乘法/加法分解（同时手动演示
       趋势(居中移动平均)与季节指数，便于理解“效应”到底是什么）。
    3. 对信用卡消费额，对比“含节日”与“去除节日”两份数据的季节/残差差异。
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "timeseries")


def plot_decomp(decomp, title, out_name):
    fig, ax = plt.subplots(4, 1, figsize=(11, 9))
    ax[0].plot(decomp.observed); ax[0].set_ylabel("原始")
    ax[1].plot(decomp.trend); ax[1].set_ylabel("趋势")
    ax[2].plot(decomp.seasonal); ax[2].set_ylabel("季节")
    ax[3].plot(decomp.resid); ax[3].set_ylabel("残差")
    ax[0].set_title(title)
    for a in ax: a.grid(alpha=0.3)
    return save_fig(fig, out_name, OUT)


def main():
    section("1. AirPassengers 月度序列 + 乘法分解")
    air = pd.read_csv(os.path.join(DATA_LEVEL1, "timeseries", "AirPassengers.csv"))
    air["DATE"] = pd.to_datetime(air["DATE"], format="%m/%d/%Y")
    air = air.set_index("DATE").asfreq("MS")["AIR"]
    print(air.head(), "\n...")
    decomp_m = seasonal_decompose(air, model="multiplicative", period=12)
    p1 = plot_decomp(decomp_m, "AirPassengers 乘法分解 (趋势×季节×残差)", "air_multiplicative.png")
    print("已保存：", p1)
    # 加法分解（对比）
    decomp_a = seasonal_decompose(air, model="additive", period=12)
    p1b = plot_decomp(decomp_a, "AirPassengers 加法分解", "air_additive.png")
    print("已保存：", p1b)

    section("2. 手动演示：趋势(12期居中MA)与季节指数")
    trend12 = air.rolling(12, center=True, min_periods=6).mean()
    detrended = air / trend12                      # 乘法去趋势
    month = detrended.index.month
    seas_index = detrended.groupby(month).mean()    # 各月平均 = 季节指数
    print("12 个月季节指数（乘法）：")
    print(seas_index.round(4))
    fig, ax = plt.subplots(figsize=(9, 4))
    seas_index.plot(marker="o", ax=ax, color="#d62728")
    ax.axhline(1.0, color="gray", ls="--")
    ax.set_title("AirPassengers 月度季节指数（>1 为旺季）")
    ax.set_xlabel("月份"); ax.set_ylabel("季节指数")
    p2 = save_fig(fig, "air_seasonal_index.png", OUT)
    print("已保存：", p2)

    section("3. 信用卡消费额：含节日 vs 去除节日")
    hol = pd.read_excel(os.path.join(DATA_LEVEL1, "timeseries", "信用卡消费额_含节日.xlsx"))
    nohol = pd.read_excel(os.path.join(DATA_LEVEL1, "timeseries", "信用卡消费额_去除节日.xlsx"))
    # 兼容表头（含节日: date/amount；去除节日: 日期/信用卡消费额）
    hcol = "amount" if "amount" in hol.columns else hol.columns[1]
    ncol = "信用卡消费额" if "信用卡消费额" in nohol.columns else nohol.columns[1]
    h = pd.Series(hol[hcol].astype(float).values)
    n = pd.Series(nohol[ncol].astype(float).values)
    dh = seasonal_decompose(h, model="additive", period=7)
    dn = seasonal_decompose(n, model="additive", period=7)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    ax[0].plot(dh.seasonal); ax[0].set_title("含节日：周度季节效应")
    ax[1].plot(dn.seasonal); ax[1].set_title("去除节日：周度季节效应")
    p3 = save_fig(fig, "creditcard_seasonal_compare.png", OUT)
    print("已保存：", p3)
    # 节日对残差(不规则效应)的影响：含节日的残差波动更大
    print(f"含节日 残差 std = {dh.resid.std():.2f}；"
          f"去除节日 残差 std = {dn.resid.std():.2f}")
    print("结论：节日会放大不规则效应(残差)的波动，去除节日后序列更平滑，"
          "更利于捕捉真实趋势与周期。")


if __name__ == "__main__":
    main()
