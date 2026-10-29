#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1章 数据分析思维 —— 维度模型（星型 Schema）Python 迁移
=========================================================
原教材用 Power BI 演示「商场销售数据表」的维度建模思想：
把一张大而全的事实表与若干维度表（品牌 / 类别 / 城市 / 日期）解耦，
通过主键关联构成“星型模型”，从而支持按任意维度切片、钻取。

本脚本用 pandas 复刻这一思想：
1. 读取事实表（产品销售数据表）与各维度表；
2. 通过维度主键做 join，重建星型模型；
3. 演示多维分析：按城市 / 品牌 / 类别 / 时间（年-季-月）聚合销售额；
4. 输出趋势、构成、对比图，等价于 Power BI 里的可视化看板。

运行：python 02_dimension_model.py
依赖：pandas, matplotlib, openpyxl
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import section, save_fig, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "dimension_model", "商场销售数据表.xlsx")
OUT = os.path.join(os.path.dirname(__file__), "output")


def main():
    section("1. 读取维度表与事实表（星型模型构件）")
    brand = pd.read_excel(SRC, sheet_name="品牌")
    category = pd.read_excel(SRC, sheet_name="类别")
    city = pd.read_excel(SRC, sheet_name="商铺城市")
    product = pd.read_excel(SRC, sheet_name="产品信息表")
    fact = pd.read_excel(SRC, sheet_name="产品销售数据表")
    date_dim = pd.read_excel(SRC, sheet_name="日期表")

    print("品牌维度   :", brand.shape, "->", list(brand.columns))
    print("类别维度   :", category.shape, "->", list(category.columns))
    print("城市维度   :", city.shape, "->", list(city.columns))
    print("产品维度   :", product.shape, "->", list(product.columns))
    print("日期维度   :", date_dim.shape, "->", list(date_dim.columns))
    print("事实表     :", fact.shape, "->", list(fact.columns))

    section("2. 维度建模：事实表 ↔ 维度表 关联（星型连接）")
    # 事实表已经带有 品牌名称/产品类别/商铺城市/产品编号，可与维度表对齐；
    # 以“产品编号”为桥梁并入产品主数据（采购单价），形成星型展开宽表
    fact = fact.merge(
        product[["产品编号", "采购单价"]],
        on="产品编号", how="left",
    )
    # 品牌维度通过“品牌名称”关联企业信息（星型：事实 ↔ 品牌维度）
    fact = fact.rename(columns={"品牌名称": "品牌"})
    brand = brand.rename(columns={"品牌名称": "品牌"})
    fact = fact.merge(brand, on="品牌", how="left")
    fact = fact.merge(city, on="商铺城市", how="left")
    fact = fact.merge(date_dim, left_on="订单日期", right_on="日期", how="left")

    fact["销售额"] = pd.to_numeric(fact["销售额"], errors="coerce")
    fact["销售数量"] = pd.to_numeric(fact["销售数量"], errors="coerce")
    print("星型展开后宽表维度键：", [c for c in fact.columns
          if c in ("品牌", "城市编码", "年度", "季度", "月", "产品类别")])
    print("总销售额(万元): %.2f" % (fact["销售额"].sum() / 1e4))

    section("3. 多维分析 · 按城市销售额（对比 + 构成）")
    by_city = (fact.groupby("商铺城市")["销售额"].sum()
               .sort_values(ascending=False) / 1e4)
    print(by_city.round(2).to_string())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(by_city.index, by_city.values, color="#4C72B0")
    ax.set_title("各商铺城市销售额对比（万元）")
    ax.set_ylabel("销售额(万元)")
    plt.xticks(rotation=30, ha="right")
    p1 = save_fig(fig, "dim_by_city.png", OUT)
    print("已保存:", p1)

    section("4. 多维分析 · 按品牌销售额（构成）")
    by_brand = (fact.groupby("品牌")["销售额"].sum()
                .sort_values(ascending=False) / 1e4)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(by_brand.values, labels=by_brand.index, autopct="%1.1f%%",
           startangle=90, colors=plt.cm.Set2.colors)
    ax.set_title("各品牌销售额构成")
    p2 = save_fig(fig, "dim_by_brand.png", OUT)
    print("已保存:", p2)

    section("5. 多维分析 · 按产品类别（构成）")
    by_cat = (fact.groupby("产品类别")["销售额"].sum()
              .sort_values(ascending=False) / 1e4)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(by_cat.index, by_cat.values, color="#55A868")
    ax.set_title("各产品类别销售额（万元）")
    ax.set_xlabel("销售额(万元)")
    p3 = save_fig(fig, "dim_by_category.png", OUT)
    print("已保存:", p3)

    section("6. 多维分析 · 年度-季度销售额趋势（钻取时间维度）")
    by_q = (fact.groupby(["年度", "季度"])["销售额"].sum() / 1e4).reset_index()
    by_q["期间"] = by_q["年度"].astype(str) + "-Q" + by_q["季度"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(by_q["期间"], by_q["销售额"], marker="o", color="#C44E52")
    ax.set_title("年度-季度销售额趋势（万元）")
    ax.set_ylabel("销售额(万元)")
    plt.xticks(rotation=45, ha="right")
    p4 = save_fig(fig, "dim_quarter_trend.png", OUT)
    print("已保存:", p4)

    section("7. 多维分析 · 城市 × 年度 透视（交叉对比）")
    pivot = pd.pivot_table(fact, values="销售额", index="商铺城市",
                           columns="年度", aggfunc="sum") / 1e4
    pivot = pivot.round(1)
    print(pivot.to_string())

    section("结论")
    print("维度模型的核心价值：事实表只存“度量+外键”，分析维度交给维度表。")
    print("上述所有切片（城市/品牌/类别/时间）都只需对宽表做 groupby，")
    print("等价于 Power BI 里拖拽字段生成的可视化，且可复现、可批处理。")


if __name__ == "__main__":
    main()
