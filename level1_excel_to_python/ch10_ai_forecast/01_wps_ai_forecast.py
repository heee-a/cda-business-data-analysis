#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第10章 —— WPS 自带 AI 时间序列预测（Python 迁移）
==================================================
教材用 WPS「自带 AI 功能」对两条序列做预测：
  · AirPassengers（1949-1960 月度国际航线客流，144 个历史点 → 预测未来 24 个月）
  · 信用卡消费额（日度，31 个历史点 → 预测未来 7 天，WPS 内部用 SARIMA）

本脚本用纯 numpy/pandas 手写 **Holt-Winters 三参数指数平滑**（水平/趋势/季节），
复刻“AI 预测”背后的统计逻辑，并把预测结果与教材 WPS 输出直接对比：

1. 解析教材文件中的历史值列与 WPS 预测列（Excel 日期序列号转日期）；
2. 用网格搜索在历史序列上确定最优 (alpha, beta, gamma)；
3. 向前多步预测，计算与 WPS 结果的 MAE / RMSE；
4. 输出对照表与可视化。

用纯 numpy 实现而不依赖 statsmodels，是为了让脚本在仅装了 pandas/numpy/matplotlib
的环境里也能直接运行（statsmodels 仍可用于进阶模型，见 requirements.txt）。

运行：python 01_wps_ai_forecast.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import section, save_fig, excel_serial_to_datetime, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "forecast")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def holt_winters(y, m, alpha, beta, gamma, h):
    """Holt-Winters 加法模型（加法趋势 + 加法季节），返回未来 h 期预测。

    状态方程：
      level_t = alpha * (y_t - s_{t-m}) + (1-alpha) * (level_{t-1} + trend_{t-1})
      trend_t = beta  * (level_t - level_{t-1}) + (1-beta) * trend_{t-1}
      season_t = gamma * (y_t - level_t) + (1-gamma) * season_{t-m}
      forecast_{t+k} = level_t + k*trend_t + season_{t-m+((k-1) mod m)+1}
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    # 初始化：用前 2 个完整季节去季节化后估计水平与趋势
    n_init = 2 * m
    season = np.zeros(n_init + n)
    for i in range(m):
        season[i] = np.mean(y[i::m]) - np.mean(y)
    level = np.mean(y[:m] - season[:m])
    trend = (np.mean(y[m:2*m] - season[m:2*m]) - level) / m
    for t in range(n_init, n):
        prev_level = level
        level = alpha * (y[t] - season[t - m]) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        season[t] = gamma * (y[t] - level) + (1 - gamma) * season[t - m]
    last_season = np.array([season[n - m + j] for j in range(m)])
    return np.array([level + k * trend + last_season[(k - 1) % m] for k in range(1, h + 1)])


def fit_best(y, m, h, wps):
    """在历史序列上做滚动起点网格搜索，用与 WPS 同长度的回测误差选超参。"""
    best = None
    for alpha in (0.2, 0.4, 0.6, 0.8):
        for beta in (0.05, 0.2, 0.4):
            for gamma in (0.05, 0.2, 0.4, 0.6):
                # 回测：用前 n-h 个点预测后 h 个真实点，评估泛化误差
                train = y[:len(y) - h]
                actual = y[len(y) - h:]
                if len(train) < 2 * m + 1:
                    continue
                pred = holt_winters(train, m, alpha, beta, gamma, h)
                err = np.sqrt(np.mean((pred - actual) ** 2))
                if best is None or err < best[0]:
                    best = (err, alpha, beta, gamma)
    err, alpha, beta, gamma = best or (0, 0.4, 0.2, 0.2)
    forecast = holt_winters(y, m, alpha, beta, gamma, h)
    return forecast, (alpha, beta, gamma), err


def compare(name, rel, m, h, out_png):
    section(name)
    df = pd.read_csv(os.path.join(SRC, rel))
    df["date"] = excel_serial_to_datetime(df["date_serial"])
    hist = pd.to_numeric(df["history"], errors="coerce").dropna().values
    wps = pd.to_numeric(df["wps_forecast"], errors="coerce").dropna().values
    print(f"历史 {len(hist)} 点（{df['date'].min().date()} ~ {df.loc[df['history'].notna(),'date'].max().date()}），"
          f"WPS 预测 {len(wps)} 点")
    if len(wps) != h:
        h = len(wps)
    pred, (alpha, beta, gamma), backtest_rmse = fit_best(hist, m, h, wps)
    mae = np.mean(np.abs(pred - wps))
    rmse = np.sqrt(np.mean((pred - wps) ** 2))
    print(f"最优参数 alpha={alpha}, beta={beta}, gamma={gamma}；回测 RMSE={backtest_rmse:.2f}")
    print(f"与 WPS 预测对比：MAE={mae:.2f}, RMSE={rmse:.2f}")
    cmp = pd.DataFrame({
        "date": df.loc[df["wps_forecast"].notna(), "date"].values,
        "WPS预测": wps,
        "Python_HoltWinters": np.round(pred, 2),
        "差异": np.round(pred - wps, 2),
    })
    print(cmp.to_string(index=False))
    cmp.to_csv(os.path.join(OUT, out_png.replace(".png", ".csv")), index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["date"], df["history"], label="历史数据", color="#4C72B0")
    fw = df.loc[df["wps_forecast"].notna(), "date"]
    ax.plot(fw, wps, "o--", label="WPS AI 预测", color="#C44E52")
    ax.plot(fw, pred, "s--", label="Python Holt-Winters", color="#55A868")
    ax.axvline(df.loc[df["history"].notna(), "date"].max(), color="gray", linestyle=":")
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    print("已保存:", save_fig(fig, out_png, OUT))
    return mae, rmse


def main():
    os.makedirs(OUT, exist_ok=True)
    compare("案例1 航空客流 AirPassengers（月度，季节周期=12）", "air_wps.csv", 12, 24, "air_forecast_compare.png")
    compare("案例2 信用卡消费额（日度，季节周期=7）", "credit_wps.csv", 7, 7, "credit_forecast_compare.png")
    print("\n结果已输出到：", OUT)


if __name__ == "__main__":
    main()
