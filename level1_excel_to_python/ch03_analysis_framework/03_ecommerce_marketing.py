#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3章 商业数据分析框架 —— 案例3：电商平台的精准营销（Python 迁移）
================================================================
原教材（Excel）用“电商精准营销”案例演示：从客户交易明细出发，
做用户画像与精准营销洞察：年龄分布、支付方式偏好、品类偏好、
各月销售趋势、以及简单的客户价值分层（RFM 思想的轻量版）。

本脚本用 pandas + matplotlib 复刻：
1. 趋势分析：月度销售额/订单量走势；
2. 分布分析：客户年龄分布；
3. 构成分析：支付方式占比、品类销量占比；
4. 用户分层：按消费额与频次做轻量 RFM 分群，支撑精准营销。

运行：python 03_ecommerce_marketing.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import section, save_fig, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "ch03_framework", "电商精准营销.xlsx")
OUT = os.path.join(os.path.dirname(__file__), "output")


def main():
    section("1. 读取客户交易明细")
    df = pd.read_excel(SRC, sheet_name="原始数据")
    df["数量"] = pd.to_numeric(df["数量"], errors="coerce")
    # 估算客单价/销售额（原始数据无金额列，用 数量 作为消费强度代理，或按品类计数）
    print("数据量:", df.shape, "列:", list(df.columns))
    print(df.head(3).to_string())

    section("2. 趋势分析：月度订单量与消费强度走势")
    df["年月"] = df["订单年份"].astype(str) + "-" + df["订单月份"].astype(str).str.zfill(2)
    monthly = df.groupby("年月").agg(订单量=("客户ID", "count"),
                                    消费数量=("数量", "sum")).reset_index()
    monthly = monthly.sort_values("年月")
    print(monthly.tail(6).to_string(index=False))
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(monthly["年月"], monthly["订单量"], marker="o", color="#4C72B0", label="订单量")
    ax1.set_ylabel("订单量", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(monthly["年月"], monthly["消费数量"], marker="s", color="#C44E52", label="消费数量")
    ax2.set_ylabel("消费数量", color="#C44E52")
    ax1.set_title("电商精准营销：月度订单量 vs 消费数量趋势")
    plt.xticks(rotation=60, ha="right")
    p1 = save_fig(fig, "ecom_trend.png", OUT)
    print("已保存:", p1)

    section("3. 分布分析：客户年龄分布")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["年龄"].dropna(), bins=20, color="#55A868", edgecolor="white")
    ax.set_title("客户年龄分布")
    ax.set_xlabel("年龄")
    ax.set_ylabel("人数")
    p2 = save_fig(fig, "ecom_age_dist.png", OUT)
    print("已保存:", p2)

    section("4. 构成分析：支付方式 与 品类 偏好")
    pay = df.groupby("支付方式")["客户ID"].count().sort_values(ascending=False)
    cat = df.groupby("品类")["数量"].sum().sort_values(ascending=False)
    print("支付方式:\n", pay.to_string())
    print("\n品类销量:\n", cat.to_string())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].pie(pay.values, labels=pay.index, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("支付方式构成")
    axes[1].barh(cat.index, cat.values, color="#8172B3")
    axes[1].set_title("各品类销量")
    axes[1].set_xlabel("销量")
    fig.tight_layout()
    p3 = save_fig(fig, "ecom_pay_category.png", OUT)
    print("已保存:", p3)

    section("5. 用户分层（轻量 RFM → 精准营销人群）")
    # 以客户为粒度：R=最近消费月份(越大越近)，F=订单频次，M=消费数量
    cust = df.groupby("客户ID").agg(
        最近月份=("订单月份", "max"),
        频次=("客户ID", "count"),
        消费量=("数量", "sum"),
    ).reset_index()
    # 二值化分群：频次、消费量均高于中位数 -> 高价值；否则普通/低活跃
    f_med, m_med = cust["频次"].median(), cust["消费量"].median()
    def seg(r):
        if r["频次"] >= f_med and r["消费量"] >= m_med:
            return "高价值客户"
        if r["频次"] < f_med and r["消费量"] < m_med:
            return "低活跃客户"
        return "潜力客户"
    cust["人群"] = cust.apply(seg, axis=1)
    seg_cnt = cust["人群"].value_counts()
    print(seg_cnt.to_string())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(seg_cnt.index, seg_cnt.values, color=["#C44E52", "#4C72B0", "#55A868"])
    ax.set_title("精准营销人群分层（轻量 RFM）")
    ax.set_ylabel("客户数")
    plt.xticks(rotation=15)
    p4 = save_fig(fig, "ecom_segment.png", OUT)
    print("已保存:", p4)

    section("结论")
    print("精准营销的本质：用交易明细刻画用户画像（年龄/支付/品类偏好），")
    print("再用 RFM 式分层圈定高价值/潜力/低活跃人群，对应差异化运营策略。")


if __name__ == "__main__":
    main()
