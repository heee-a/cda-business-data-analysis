#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 04：同期群分析（Cohort Analysis，对应 1-4 同期群分析）
==================================================================
【原 Excel/Power BI 实现内涵】
    同期群分析按“注册批次(同期群)”分组，观察每批用户在注册后第 N 天的
    留存/活跃情况，用于判断产品黏性是否随时间改善。教材数据给出每个用户的
    注册日期、登录日期、以及“第几日”。

【Python 迁移要点】
    1. 读「用户登录明细」，把注册日期/登录日期转成日期，计算注册后第几天。
    2. 以“注册月份”作为同期群，以“注册后天数”作为时间偏移。
    3. 透视出 [同期群 × 天数] 的活跃用户数，再换算成留存率矩阵。
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, excel_serial_to_datetime, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "cohort", "1-4同期群分析-参考.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "cohort")


def main():
    section("1. 读取用户登录明细")
    df = pd.read_excel(SRC, sheet_name="用户登录明细")
    df["注册日期"] = excel_serial_to_datetime(df["注册日期"])
    df["登录日期"] = excel_serial_to_datetime(df["登录日期"])
    df["offset"] = (df["登录日期"] - df["注册日期"]).dt.days
    df["cohort"] = df["注册日期"].dt.to_period("M").astype(str)
    print(df.head())
    print("注册批次(前10)：", sorted(df["cohort"].unique())[:10])

    section("2. 构建 同期群 × 注册后天数 的活跃用户数")
    # 每个 (cohort, offset) 去重统计用户数
    cohort_data = (df.groupby(["cohort", "offset"])["用户编号"].nunique()
                     .reset_index().rename(columns={"用户编号": "活跃用户数"}))
    # 每个 cohort 的初始规模（offset==0 或 最早一天）作为分母
    base = (df.groupby("cohort")["用户编号"].nunique())
    pivot = cohort_data.pivot(index="cohort", columns="offset", values="活跃用户数")
    # 仅保留 0..30 天便于展示
    cols = [c for c in pivot.columns if 0 <= c <= 30]
    pivot = pivot[sorted(cols)]
    print("活跃用户数透视表(前5行)：")
    print(pivot.head())

    section("3. 换算留存率(%)")
    retention = pivot.div(base, axis=0) * 100
    retention = retention.round(1)
    print("留存率矩阵(前5行)：")
    print(retention.head())

    section("4. 可视化——留存热力图")
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(retention.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(retention.columns)))
    ax.set_xticklabels(retention.columns)
    ax.set_yticks(range(len(retention.index)))
    ax.set_yticklabels(retention.index, fontsize=7)
    ax.set_xlabel("注册后天数")
    ax.set_ylabel("注册批次(同期群)")
    ax.set_title("同期群留存率热力图 (%)")
    fig.colorbar(im, ax=ax, label="留存率(%)")
    p = save_fig(fig, "cohort_retention.png", OUT)
    print("已保存：", p)

    out_csv = os.path.join(OUT, "cohort_retention.csv")
    retention.to_csv(out_csv, encoding="utf-8-sig")
    print("已保存留存率表：", out_csv)


if __name__ == "__main__":
    main()
