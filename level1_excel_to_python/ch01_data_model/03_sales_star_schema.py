#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章 PowerBI 基本操作 —— 创建数据模型示例（星型模型，Python 迁移）
====================================================================
教材用「创建数据模型示例.xlsx」在 Power BI 里建立星型模型：
销售业绩表(事实) 通过 客户ID/城市编号/产品编号 关联 客户/城市/产品 三张维度表，
再用切片器按区域、企业规模、产品系列做交叉分析。本脚本用 pandas 复刻：

1. 构建星型模型：事实表 + 3 张维度表，外键连接成分析宽表；
2. 按区域/企业规模/产品类型/产品系列多维汇总销售额；
3. 交叉透视：区域 × 企业规模 销售矩阵；
4. Top N 客户与产品；
5. 可视化：区域销售、产品系列销售。

运行：python 03_sales_star_schema.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
from common import section, save_fig, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "sales_model")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "sales_model")


def main():
    os.makedirs(OUT, exist_ok=True)
    sales = pd.read_csv(os.path.join(SRC, "sales.csv"))
    customers = pd.read_csv(os.path.join(SRC, "customers.csv"))
    cities = pd.read_csv(os.path.join(SRC, "cities.csv"))
    products = pd.read_csv(os.path.join(SRC, "products.csv"))
    amount_col = "金额（$M）"

    section("1. 星型模型：事实表 + 维度表")
    for name, df in [("销售业绩(事实)", sales), ("客户", customers), ("城市", cities), ("产品", products)]:
        print(f"  {name}: {df.shape}")
    # 事实表分别 LEFT JOIN 三张维表，等价于 Power BI 建立关系后的展平
    star = (sales.merge(customers, on="客户ID", how="left")
                 .merge(cities, on="城市编号", how="left")
                 .merge(products, on="产品编号", how="left"))
    star[amount_col] = pd.to_numeric(star[amount_col], errors="coerce")
    print(f"\n关联后分析宽表：{star.shape}，总销售额 {star[amount_col].sum():,.1f} $M")
    print(star.head(3).to_string(index=False))

    section("2. 多维汇总：区域 / 企业规模 / 产品类型 / 产品系列")
    for dim in ["区域名", "企业规模", "产品类型", "产品系列", "企业性质"]:
        agg = star.groupby(dim)[amount_col].sum().sort_values(ascending=False).round(1)
        print(f"\n按{dim}：\n{agg.head(10).to_string()}")

    section("3. 交叉透视：区域 × 企业规模")
    pivot = pd.pivot_table(star, values=amount_col, index="区域名", columns="企业规模",
                           aggfunc="sum", fill_value=0).round(1)
    print(pivot.to_string())
    pivot.to_csv(os.path.join(OUT, "region_scale_pivot.csv"), encoding="utf-8-sig")

    section("4. Top N 客户与产品")
    top_cust = star.groupby("客户名称")[amount_col].sum().sort_values(ascending=False).head(10).round(1)
    print("Top10 客户：\n", top_cust.to_string())
    top_prod = star.groupby("产品")[amount_col].sum().sort_values(ascending=False).head(10).round(1)
    print("\nTop10 产品：\n", top_prod.to_string())

    section("5. 可视化")
    region = star.groupby("区域名")[amount_col].sum().sort_values(ascending=False)
    series = star.groupby("产品系列")[amount_col].sum().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(region.index, region.values, color="#4C72B0")
    axes[0].set_title("各区域销售额($M)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(series.index, series.values, color="#55A868")
    axes[1].set_title("各产品系列销售额($M)")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    print("已保存:", save_fig(fig, "star_schema_summary.png", OUT))
    print("\n结果已输出到：", OUT)


if __name__ == "__main__":
    main()
