#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章 PowerBI 基本操作 —— 案例1 分省销售分析（Python 迁移）
============================================================
教材用「分省销售分析.pbix」演示一个典型的小型销售数据市集：
订单表(事实) + 订单明细(事实) + 商品表/商品类型表/雇员表(维度)。
本脚本用 pandas 复刻其分析逻辑：

1. 事实×维度关联：订单→明细→商品→商品类别、订单→雇员；
2. 分省销售：各省销售额、订单数；
3. 商品类别构成：各大类销售额占比；
4. 雇员业绩：各雇员销售额与毛利；
5. 月度趋势与可视化。

注：原始数据列名「产口ID」为教材笔误，实际指「产品ID」，脚本中已纠正。
运行：python 01_province_sales.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
from common import section, save_fig, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "province_sales")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    os.makedirs(OUT, exist_ok=True)
    orders = pd.read_csv(os.path.join(SRC, "orders.csv"))
    detail = pd.read_csv(os.path.join(SRC, "order_details.csv"))
    products = pd.read_csv(os.path.join(SRC, "products.csv"))
    categories = pd.read_csv(os.path.join(SRC, "categories.csv"))
    employees = pd.read_csv(os.path.join(SRC, "employees.csv"))

    section("1. 数据市集概览")
    for name, df in [("订单表", orders), ("订单明细", detail), ("商品", products),
                     ("商品类型", categories), ("雇员", employees)]:
        print(f"  {name}: {df.shape}, 列={list(df.columns)}")
    # 列 cleansing：教材明细表「产口ID」= 产品ID
    detail = detail.rename(columns={"产口ID": "产品ID"})
    for c in ["销售金额", "销售成本"]:
        detail[c] = pd.to_numeric(detail[c], errors="coerce")
    detail["毛利"] = detail["销售金额"] - detail["销售成本"]
    orders["OrderDate"] = pd.to_datetime(orders["OrderDate"], errors="coerce")

    section("2. 明细 × 商品 × 商品类别 × 订单（星型展开）")
    fact = (detail.merge(products, left_on="产品ID", right_on="ProductID", how="left")
                  .merge(categories, on="CategoryID", how="left")
                  .merge(orders, left_on="订单ID", right_on="OrderID", how="left"))
    fact = fact.rename(columns={"ProductName": "商品名称", "CategoryName": "商品类别",
                                "ShipProvince": "省份", "EmployeeID": "雇员ID"})
    print(f"展开后事实宽表：{fact.shape}")
    print(fact[["订单ID", "商品名称", "商品类别", "销售金额", "毛利", "省份"]].head(5).to_string(index=False))

    section("3. 分省销售：销售额、订单数、毛利")
    by_prov = fact.groupby("省份").agg(销售额=("销售金额", "sum"), 毛利=("毛利", "sum"),
                                       明细行数=("订单ID", "size")).sort_values("销售额", ascending=False)
    by_prov["订单数"] = fact.groupby("省份")["订单ID"].nunique()
    print(by_prov.round(1).to_string())
    by_prov.to_csv(os.path.join(OUT, "by_province.csv"), encoding="utf-8-sig")

    section("4. 商品类别构成")
    by_cat = fact.groupby("商品类别").agg(销售额=("销售金额", "sum"), 毛利=("毛利", "sum")).sort_values("销售额", ascending=False)
    by_cat["占比%"] = (by_cat["销售额"] / by_cat["销售额"].sum() * 100).round(1)
    print(by_cat.round(1).to_string())

    section("5. 雇员业绩")
    emp = (fact.merge(employees, left_on="雇员ID", right_on="EmployeeID", how="left")
               .groupby("FullName").agg(销售额=("销售金额", "sum"), 毛利=("毛利", "sum"),
                                        订单数=("订单ID", "nunique")).sort_values("销售额", ascending=False))
    print(emp.round(1).to_string())

    section("6. 月度趋势与可视化")
    orders["月份"] = orders["OrderDate"].dt.to_period("M").astype(str)
    monthly = orders.groupby("月份").size()
    print("每月订单数：\n", monthly.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    by_prov["销售额"].head(12).plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_title("各省销售额 Top12")
    axes[0].invert_yaxis()
    by_cat["销售额"].plot(kind="pie", ax=axes[1], autopct="%1.0f%%", startangle=90)
    axes[1].set_title("商品类别销售额构成")
    axes[1].set_ylabel("")
    fig.tight_layout()
    print("已保存:", save_fig(fig, "province_sales.png", OUT))
    fact.to_csv(os.path.join(OUT, "sales_fact_wide.csv"), index=False, encoding="utf-8-sig")
    print("\n结果已输出到：", OUT)


if __name__ == "__main__":
    main()
