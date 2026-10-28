#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 02：波士顿矩阵（BCG Matrix，对应 1-1 波士顿矩阵）
==================================================================
【原 Excel/Power BI 实现内涵】
    给定各品牌的销售流水，教材用波士顿矩阵把品牌按两个维度定位：
      · 市场增长率（业务是否处于高速增长赛道）
      · 相对市场份额（品牌在该品类里的相对地位）
    组合出 4 类：明星(star) / 现金牛(cash cow) / 问题(question) / 瘦狗(dog)。
    Excel 做法：透视表算各品牌销售额与占比，再用公式计算增长率与相对份额。

【Python 迁移要点】
    1. 读「销售表」流水，按 品牌 聚合销售额（订单金额求和）。
    2. 以中位日期为界把时间切成两期，算每个品牌的市场增长率。
    3. 算各品牌市场占有率，并与“市场平均占有率 / 最大占有率”比较得相对份额。
    4. 按 (增长率高低, 相对份额高低) 四象限归类并画图。
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, excel_serial_to_datetime, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "boston", "1-1波士顿矩阵-参考.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "boston")


def main():
    section("1. 读取销售流水")
    df = pd.read_excel(SRC, sheet_name="销售表")
    df["销售日期"] = excel_serial_to_datetime(df["销售日期"])
    df["订单金额"] = pd.to_numeric(df["订单金额"], errors="coerce")
    print(f"记录数：{len(df):,}，品牌数：{df['品牌'].nunique()}")
    print(df.head())

    section("2. 各品牌销售额与市场占有率")
    brand = df.groupby("品牌")["订单金额"].sum().sort_values(ascending=False)
    total = brand.sum()
    share = brand / total
    print("品牌平均市场占有率 ≈", round(share.mean(), 4))
    print(brand.head(10))

    section("3. 以中位日期为界算各品牌市场增长率")
    mid = df["销售日期"].median()
    p1 = df[df["销售日期"] <= mid].groupby("品牌")["订单金额"].sum()
    p2 = df[df["销售日期"] > mid].groupby("品牌")["订单金额"].sum()
    growth = ((p2 - p1) / p1).fillna(0)
    market_growth = (p2.sum() - p1.sum()) / p1.sum()  # 总市场环比增长率
    print("总市场环比增长率 ≈", round(market_growth, 4))

    section("4. 构建波士顿矩阵四象限")
    rel_share = share / share.max()           # 相对市场份额(以最大者为1)
    share_thr = share.mean()                   # 份额高于平均=高份额
    growth_thr = market_growth                 # 增长率高于市场平均=高增
    matrix = pd.DataFrame({"销售额": brand, "市场占有率": share,
                           "相对份额": rel_share, "增长率": growth})
    def classify(r):
        hi_g = r["增长率"] > growth_thr
        hi_s = r["市场占有率"] > share_thr
        if hi_g and hi_s: return "明星业务(高增长高份额)"
        if hi_g and not hi_s: return "问题业务(高增长低份额)"
        if (not hi_g) and hi_s: return "现金牛业务(低增长高份额)"
        return "瘦狗业务(低增长低份额)"
    matrix["象限"] = matrix.apply(classify, axis=1)
    print(matrix.round(4))

    section("5. 可视化")
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"明星业务(高增长高份额)": "#2ca02c", "问题业务(高增长低份额)": "#ff7f0e",
              "现金牛业务(低增长高份额)": "#1f77b4", "瘦狗业务(低增长低份额)": "#d62728"}
    for quad, sub in matrix.groupby("象限"):
        ax.scatter(sub["相对份额"], sub["增长率"], s=sub["销售额"] / brand.max() * 1500 + 30,
                   c=colors.get(quad, "#888"), alpha=0.6, label=quad)
        for name, row in sub.iterrows():
            ax.annotate(name, (row["相对份额"], row["增长率"]), fontsize=8,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(growth_thr, color="gray", ls="--")
    ax.axvline(share_thr / share.max(), color="gray", ls="--")
    ax.set_xlabel("相对市场份额")
    ax.set_ylabel("市场增长率")
    ax.set_title("波士顿矩阵（气泡大小=品牌销售额）")
    ax.legend(loc="upper right", fontsize=8)
    p = save_fig(fig, "boston_matrix.png", OUT)
    print("已保存：", p)

    out_csv = os.path.join(OUT, "boston_result.csv")
    matrix.to_csv(out_csv, encoding="utf-8-sig")
    print("已保存结果表：", out_csv)


if __name__ == "__main__":
    main()
