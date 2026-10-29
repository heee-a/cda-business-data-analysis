#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第5章 业务数据分析 —— 综合案例：银行产品多维销售分析（Python 迁移）
================================================================
原教材用 Power BI 搭建“银行案例”业务看板。底层是银行产品销售的客户级数据，
配合 风险偏好 / 品类 / 学历 三张维度说明表。

本脚本用 pandas 复刻多维销售分析：
1. 品类销售构成：各产品品类 购买金额 占比；
2. 购买目的构成；
3. 客户画像：性别构成、年龄分布；
4. 选择本行原因构成（渠道/获客归因视角）；
5. 风险偏好 vs 平均收益；
6. 家庭年收入 vs 购买金额（关系分析）。

运行：python 02_bank_sales_analysis.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
from common import section, save_fig, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "bank_sales", "银行案例数据.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "bank")


def main():
    section("1. 读取销售事实与维度说明表")
    sales = pd.read_excel(SRC, sheet_name="销售数据")
    risk = pd.read_excel(SRC, sheet_name="风险偏好说明表")
    cat = pd.read_excel(SRC, sheet_name="品类说明表")
    edu = pd.read_excel(SRC, sheet_name="学历说明表")
    print("销售数据:", sales.shape, list(sales.columns))
    for name, df in [("风险", risk), ("品类", cat), ("学历", edu)]:
        print(f"  {name}说明: {df.shape} -> {list(df.columns)}")

    # 类型规整
    for c in ["家庭年收入", "年龄", "购买金额", "期限", "收益"]:
        sales[c] = pd.to_numeric(sales[c], errors="coerce")
    # 维度 join
    sales = sales.merge(cat, on="品类编号", how="left")
    sales = sales.merge(risk, on="风险编号", how="left")
    sales = sales.merge(edu, on="学历编号", how="left")
    print("合并后维度:", [c for c in sales.columns if c in
          ("品类名称", "风险说明", "学历名称")])

    section("2. 品类销售构成")
    by_cat = sales.groupby("品类名称")["购买金额"].sum().sort_values(ascending=False) / 1e4
    print(by_cat.round(2).to_string())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(by_cat.values, labels=by_cat.index, autopct="%1.1f%%", startangle=90,
           colors=plt.cm.Set2.colors)
    ax.set_title("各产品品类购买金额构成（万元）")
    p1 = save_fig(fig, "bank_category.png", OUT)
    print("已保存:", p1)

    section("3. 购买目的构成")
    by_purp = sales.groupby("购买目的")["购买金额"].sum().sort_values(ascending=False)
    print(by_purp.round(2).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(by_purp.index.astype(str), by_purp.values / 1e4, color="#4C72B0")
    ax.set_title("购买目的构成（万元）")
    ax.set_xlabel("购买金额(万元)")
    p2 = save_fig(fig, "bank_purpose.png", OUT)
    print("已保存:", p2)

    section("4. 客户画像：性别构成 + 年龄分布")
    by_sex = sales.groupby("性别")["购买金额"].sum()
    print("性别购买金额:\n", by_sex.round(2).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].pie(by_sex.values, labels=by_sex.index, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("性别购买金额构成")
    axes[1].hist(sales["年龄"].dropna(), bins=15, color="#55A868", edgecolor="white")
    axes[1].set_title("客户年龄分布"); axes[1].set_xlabel("年龄")
    fig.tight_layout()
    p3 = save_fig(fig, "bank_profile.png", OUT)
    print("已保存:", p3)

    section("5. 选择本行原因构成（获客渠道归因视角）")
    by_reason = sales.groupby("选择本行的原因")["购买金额"].sum().sort_values(ascending=False)
    print(by_reason.round(2).to_string())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(by_reason.index.astype(str), by_reason.values / 1e4, color="#8172B3")
    ax.set_title("选择本行原因构成（万元）")
    ax.set_xlabel("购买金额(万元)")
    p4 = save_fig(fig, "bank_reason.png", OUT)
    print("已保存:", p4)

    section("6. 风险偏好 vs 平均收益")
    by_risk = sales.groupby("风险说明")["收益"].mean()
    print(by_risk.round(3).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_risk.index.astype(str), by_risk.values, color="#C44E52")
    ax.set_title("不同风险偏好平均收益")
    ax.set_ylabel("平均收益")
    plt.xticks(rotation=20)
    p5 = save_fig(fig, "bank_risk.png", OUT)
    print("已保存:", p5)

    section("7. 关系分析：家庭年收入 vs 购买金额")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(sales["家庭年收入"], sales["购买金额"], alpha=0.5, color="#4C72B0", s=20)
    ax.set_title("家庭年收入 vs 购买金额")
    ax.set_xlabel("家庭年收入"); ax.set_ylabel("购买金额")
    p6 = save_fig(fig, "bank_income_amount.png", OUT)
    print("已保存:", p6)

    section("结论")
    print("银行综合案例 = 在维度模型（品类/风险/学历）上对客户级销售事实做多维分析：")
    print("从品类/目的/渠道构成到客户画像与风险-收益关系，构成完整业务看板。")


if __name__ == "__main__":
    main()
