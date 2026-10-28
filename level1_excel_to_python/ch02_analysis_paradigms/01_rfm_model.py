#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 01：RFM 模型（对应教材第 2 章「分析的基础范式」— 1-2 RFM 模型）
==================================================================
【原 Excel/Power BI 实现内涵】
    教材给出一份零售交易流水（流水号、用户编号、品类、销售日期、销售金额），
    要求从「最近一次消费(Recency)、消费频率(Frequency)、消费金额(Monetary)」
    三个维度刻画每个用户，并据此把用户划分成 8 类（重要价值 / 重要保持 /
    重要发展 / 重要挽留 / 一般价值 / 一般保持 / 一般发展 / 一般挽留）。
    Excel 做法：透视表按用户汇总出 R/F/M，再用均值作阈值打标签。

【Python 迁移要点】
    1. 用 pandas 读取 xlsx 的「原始数据」工作表；Excel 日期是序列号，需转换。
    2. 以「分析时点=43680」为基准计算 R（距最近一次消费的天数）。
    3. groupby 用户编号聚合得到 F(交易笔数) 与 M(销售金额合计)。
    4. 以 R/F/M 的均值为分界，高于均值记“高”、低于记“低”，组合出 8 类。
    5. 输出每类用户数、各类的 R/F/M 均值，并绘制可视化。

运行：python 01_rfm_model.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, excel_serial_to_datetime, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "rfm", "1-2RFM模型-参考.xlsx")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "rfm")


def main():
    section("1. 读取原始交易流水")
    raw = pd.read_excel(SRC, sheet_name="原始数据")
    # 列：流水号, 用户编号, 一级品类, 二级品类, 产品描述, 会员创建日期, 销售日期, 销售金额
    raw["销售日期"] = excel_serial_to_datetime(raw["销售日期"])
    raw["会员创建日期"] = excel_serial_to_datetime(raw["会员创建日期"])
    print(f"交易记录数：{len(raw):,}，用户数：{raw['用户编号'].nunique():,}")
    print(raw.head())

    section("2. 计算 R / F / M")
    ANALYSIS_DATE = excel_serial_to_datetime(pd.Series([43680])).iloc[0]
    print("分析时点：", ANALYSIS_DATE.date())
    g = raw.groupby("用户编号")
    rfm = pd.DataFrame({
        "R": (ANALYSIS_DATE - g["销售日期"].max()).dt.days,   # 最近一次消费距今天数
        "F": g["流水号"].count(),                              # 消费频次（交易笔数）
        "M": g["销售金额"].sum(),                              # 消费金额合计
    })
    rfm = rfm.reset_index()
    print("R/F/M 均值：", rfm[["R", "F", "M"]].mean().round(2).to_dict())
    print(rfm.head())

    section("3. 以均值作阈值，划分 8 类用户")
    rmean, fmean, mmean = rfm[["R", "F", "M"]].mean()
    rfm["R标签"] = (rfm["R"] < rmean).map({True: "高", False: "低"})   # R越小越“高”
    rfm["F标签"] = (rfm["F"] > fmean).map({True: "高", False: "低"})
    rfm["M标签"] = (rfm["M"] > mmean).map({True: "高", False: "低"})

    seg_map = {
        ("高", "高", "高"): "重要价值客户",
        ("低", "高", "高"): "重要保持客户",
        ("高", "低", "高"): "重要发展客户",
        ("低", "低", "高"): "重要挽留客户",
        ("高", "高", "低"): "一般价值客户",
        ("低", "高", "低"): "一般保持客户",
        ("高", "低", "低"): "一般发展客户",
        ("低", "低", "低"): "一般挽留客户",
    }
    rfm["用户类别"] = rfm[["R标签", "F标签", "M标签"]].apply(
        lambda r: seg_map[(r["R标签"], r["F标签"], r["M标签"])], axis=1)

    seg = (rfm.groupby("用户类别")
              .agg(用户数=("用户编号", "count"),
                   R均值=("R", "mean"), F均值=("F", "mean"), M均值=("M", "mean"))
              .round(2))
    seg = seg.reindex(seg_map.values())
    print(seg)

    section("4. 可视化")
    fig, ax = plt.subplots(figsize=(9, 5))
    seg["用户数"].plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("RFM 八类用户分布")
    ax.set_xlabel("用户数")
    for i, v in enumerate(seg["用户数"]):
        ax.text(v + max(seg["用户数"]) * 0.01, i, str(int(v)), va="center")
    p1 = save_fig(fig, "rfm_segments.png", OUT)
    print("已保存：", p1)

    # 各类 R/F/M 均值对比
    fig, ax = plt.subplots(figsize=(10, 5))
    seg[["R均值", "F均值", "M均值"]].plot(kind="bar", ax=ax)
    ax.set_title("各类用户的 R / F / M 均值对比")
    ax.set_xticklabels(seg.index, rotation=30, ha="right")
    p2 = save_fig(fig, "rfm_means.png", OUT)
    print("已保存：", p2)

    # 保存结果表
    out_xlsx = os.path.join(OUT, "rfm_result.csv")
    rfm.to_csv(out_xlsx, index=False, encoding="utf-8-sig")
    print("已保存用户级结果：", out_xlsx)
    print("\n完成。核心结论：RFM 用三个最朴素的行为指标即可完成用户价值分层，"
          "是用户运营中最常用的基础分析范式之一。")


if __name__ == "__main__":
    main()
