#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 03：用户忠诚度模型（对应 1-3 用户忠诚度模型）
==================================================================
【原 Excel/Power BI 实现内涵】
    教材在 8 万行销售流水上做了两件事：
      (1) 用户生命周期判定：以“分析时点=2021-01-01、最近13周=2020-10-02”为界，
          定义 潜在用户（此前0次、此后恰好1次消费）、易流失用户（此后0次消费）。
      (2) 用户价值分层：按 消费次数(F)、消费额(M) 评分(1~3)，组合出
          低价值 / 普通 / 忠诚 用户。
    Power BI 用 DAX 度量值(sum / distinctcount) + 切片器完成，这里用 pandas 复刻。

【Python 迁移要点】
    1. 读「销售流水表」，按 会员编号 聚合。
    2. 用 消费日期 与两个时间节点比较，判定 潜在 / 易流失 / 活跃。
    3. 用分位数把 F、M 映射成 1~3 分，组合出用户类别。
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, excel_serial_to_datetime, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "loyalty", "1-3用户忠诚度模型-参考.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "loyalty")

ANALYSIS_DATE = pd.Timestamp("2021-01-01")
RECENT_13W = pd.Timestamp("2020-10-02")


def main():
    section("1. 读取销售流水表")
    df = pd.read_excel(SRC, sheet_name="销售流水表")
    df["消费日期"] = excel_serial_to_datetime(df["消费日期"])
    df["消费金额"] = pd.to_numeric(df["消费金额"], errors="coerce")
    print(f"流水行数：{len(df):,}，会员数：{df['会员编号'].nunique():,}")

    section("2. 用户生命周期判定：潜在 / 易流失 / 活跃")
    g = df.groupby("会员编号")
    user = pd.DataFrame({
        "消费次数": g["订单编号"].nunique(),
        "消费额": g["消费金额"].sum(),
    })
    before = df[df["消费日期"] < RECENT_13W].groupby("会员编号")["订单编号"].nunique()
    after = df[df["消费日期"] >= RECENT_13W].groupby("会员编号")["订单编号"].nunique()
    user["before"] = before.reindex(user.index).fillna(0)
    user["after"] = after.reindex(user.index).fillna(0)

    def life(u):
        if u["before"] == 0 and u["after"] == 1:
            return "潜在用户"
        if u["after"] == 0:
            return "易流失用户"
        return "活跃用户"
    user["生命周期"] = user.apply(life, axis=1)
    print(user["生命周期"].value_counts())

    section("3. F/M 评分(1~3 分位) 与 用户类别")
    user["F评分"] = pd.qcut(user["消费次数"], 3, labels=[1, 2, 3]).astype(int)
    user["M评分"] = pd.qcut(user["消费额"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    user["总分"] = user["F评分"] + user["M评分"]

    def category(r):
        if r["F评分"] >= 3 and r["M评分"] >= 3:
            return "忠诚用户"
        if r["F评分"] <= 1 and r["M评分"] <= 1:
            return "低价值用户"
        return "普通用户"
    user["用户类别"] = user.apply(category, axis=1)
    print(user["用户类别"].value_counts())

    section("4. 可视化")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    user["用户类别"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax[0])
    ax[0].set_title("用户价值类别占比")
    user["生命周期"].value_counts().plot.bar(ax=ax[1], color="#55a868")
    ax[1].set_title("用户生命周期分布")
    p = save_fig(fig, "loyalty.png", OUT)
    print("已保存：", p)

    out_csv = os.path.join(OUT, "loyalty_result.csv")
    user.reset_index().to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("已保存用户级结果：", out_csv)


if __name__ == "__main__":
    main()
