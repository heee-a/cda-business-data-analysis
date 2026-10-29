#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章 数据分析方法 —— 综合案例：服装零售销售分析（Python 迁移）
================================================================
原教材用 Power BI 搭建“服装代理商销售分析”看板。底层是一个零售数据市集：
事实表 零售单 / 零售单明细，维度表 店铺 / VIP / SKU / 款号 / 颜色 等。

本脚本用 pandas 复刻该看板的多维分析：
1. 销售趋势：按订单日期的月度成交金额走势；
2. 店铺对比：各店铺销售额对比 + 坪效（单位面积销售额）；
3. 会员贡献：VIP vs 非会员 成交额占比（用户价值）；
4. 品类(大类)构成：通过 SKU→款号 链路汇总各大类销售额；
5. 颜色偏好：通过 SKU→颜色 汇总各颜色销售额；
6. 折扣分析：折扣率分布及其对成交的影响。

运行：python 12_retail_sales_case.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
from common import section, save_fig, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "retail", "服装零售-原始数据.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "retail")


def main():
    section("1. 读取事实表与维度表（零售数据市集）")
    order = pd.read_excel(SRC, sheet_name="零售单")
    detail = pd.read_excel(SRC, sheet_name="零售单明细")
    store = pd.read_excel(SRC, sheet_name="店铺表")
    vip = pd.read_excel(SRC, sheet_name="vip表")
    sku = pd.read_excel(SRC, sheet_name="SKU表")
    style = pd.read_excel(SRC, sheet_name="款号表")
    color = pd.read_excel(SRC, sheet_name="颜色表")
    for name, df in [("零售单", order), ("零售单明细", detail), ("店铺", store),
                     ("VIP", vip), ("SKU", sku), ("款号", style), ("颜色", color)]:
        print(f"  {name}: {df.shape} -> {list(df.columns)[:6]}")

    # 类型规整
    for c in ["零售金额", "成交金额", "数量"]:
        order[c] = pd.to_numeric(order[c], errors="coerce")
        detail[c] = pd.to_numeric(detail[c], errors="coerce")
    order["订单日期"] = pd.to_datetime(order["订单日期"], errors="coerce")

    section("2. 销售趋势：月度成交金额")
    order["月份"] = order["订单日期"].dt.to_period("M").astype(str)
    monthly = order.groupby("月份")["成交金额"].sum()
    print(monthly.tail(6).to_string())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(monthly.index, monthly.values / 1e4, marker="o", color="#4C72B0")
    ax.set_title("服装零售：月度成交金额趋势（万元）")
    ax.set_ylabel("成交金额(万元)")
    plt.xticks(rotation=60, ha="right")
    p1 = save_fig(fig, "retail_trend.png", OUT)
    print("已保存:", p1)

    section("3. 店铺对比 + 坪效")
    so = order.merge(store, on="店铺id", how="left")
    by_store = so.groupby("店铺名称")["成交金额"].sum().sort_values(ascending=False) / 1e4
    # 坪效：成交金额 / 店铺面积
    eff = so.groupby("店铺名称").agg(金额=("成交金额", "sum"),
                                     面积=("店铺面积", "first"))
    eff["坪效(元/㎡)"] = (eff["金额"] / eff["面积"]).round(0)
    print("各店铺成交额(万元):\n", by_store.round(2).to_string())
    print("\n坪效 Top:\n", eff.sort_values("坪效(元/㎡)", ascending=False)["坪效(元/㎡)"].head(5).to_string())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(by_store.index, by_store.values, color="#55A868")
    ax.set_title("各店铺成交金额对比（万元）")
    ax.set_ylabel("成交金额(万元)")
    plt.xticks(rotation=30, ha="right")
    p2 = save_fig(fig, "retail_by_store.png", OUT)
    print("已保存:", p2)

    section("4. 会员贡献：VIP vs 非会员")
    order_v = order.merge(vip, left_on="会员id", right_on="VIPid", how="left")
    order_v["是否会员"] = order_v["VIPid"].notna().map({True: "会员", False: "非会员"})
    by_vip = order_v.groupby("是否会员")["成交金额"].sum()
    print(by_vip.round(2).to_string())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(by_vip.values, labels=by_vip.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("会员 vs 非会员 成交额贡献")
    p3 = save_fig(fig, "retail_vip.png", OUT)
    print("已保存:", p3)

    section("5. 品类(大类)构成：SKU→款号 链路")
    d = detail.merge(sku, left_on="商品id", right_on="SKUid", how="left")
    d = d.merge(style, on="款号id", how="left")
    by_cat = d.groupby("大类id")["成交金额"].sum().sort_values(ascending=False) / 1e4
    print(by_cat.round(2).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(by_cat.index.astype(str), by_cat.values, color="#8172B3")
    ax.set_title("各大类成交金额构成（万元）")
    ax.set_xlabel("成交金额(万元)")
    p4 = save_fig(fig, "retail_category.png", OUT)
    print("已保存:", p4)

    section("6. 颜色偏好：SKU→颜色 链路")
    d2 = detail.merge(sku, left_on="商品id", right_on="SKUid", how="left")
    d2 = d2.merge(color, on="颜色id", how="left")
    by_color = d2.groupby("颜色名称")["成交金额"].sum().sort_values(ascending=False).head(10) / 1e4
    print(by_color.round(2).to_string())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_color.index.astype(str), by_color.values, color="#C44E52")
    ax.set_title("Top10 颜色成交金额（万元）")
    ax.set_ylabel("成交金额(万元)")
    plt.xticks(rotation=45, ha="right")
    p5 = save_fig(fig, "retail_color.png", OUT)
    print("已保存:", p5)

    section("7. 折扣分析")
    detail["折扣率"] = pd.to_numeric(detail["折扣率"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(detail["折扣率"].dropna(), bins=20, color="#55A868", edgecolor="white")
    ax.set_title("折扣率分布")
    ax.set_xlabel("折扣率"); ax.set_ylabel("明细行数")
    p6 = save_fig(fig, "retail_discount.png", OUT)
    print("已保存:", p6)

    section("结论")
    print("服装零售综合案例 = 用维度模型（店铺/VIP/SKU/款号/颜色）对销售事实做多维切片：")
    print("趋势、店铺对比、会员价值、品类/颜色构成、折扣——与 Power BI 看板一一对应。")


if __name__ == "__main__":
    main()
