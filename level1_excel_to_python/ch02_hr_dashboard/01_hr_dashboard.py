#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章 PowerBI 函数练习 —— 人力资源看板（Python 迁移）
======================================================
教材用 Power BI + DAX 函数（CALENDAR、DATEDIF、COUNTIF、CALCULATE 等）
基于一张员工信息表搭建人力资源看板。本脚本用 pandas 复刻：

1. 人员结构：部门 / 学历 / 性别分布（COUNTIF 类统计）；
2. 年龄与司龄：出生日期、入职日期由 Excel 序列号转日期，计算年龄/司龄（DATEDIF）；
3. 在职分析：在职/离职人数与离职率，离职员工的司龄分布；
4. 职级画像：各部门平均职级、职称层级结构；
5. 可视化：部门人数、学历构成、年龄段分布的图表。

运行：python 01_hr_dashboard.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
from common import section, save_fig, excel_serial_to_datetime, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "hr", "employees.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(SRC, dtype=str)
    df.columns = ["员工工号", "姓名", "性别", "部门", "职称", "职级", "出生日期",
                  "入职日期", "离职日期", "员工状态", "文化程度"]
    section("1. 数据概览与日期解析")
    print("员工总数：", len(df))
    print("字段：", list(df.columns))
    df["职级"] = pd.to_numeric(df["职级"], errors="coerce")
    df["出生日期"] = excel_serial_to_datetime(df["出生日期"])
    df["入职日期"] = excel_serial_to_datetime(df["入职日期"])
    df["离职日期"] = excel_serial_to_datetime(df["离职日期"])
    as_of = pd.Timestamp("2022-12-31")  # 看板基准日，与教材口径一致
    df["年龄"] = ((as_of - df["出生日期"]).dt.days / 365.25).round().astype("Int64")
    df["司龄"] = (((df["离职日期"].fillna(as_of) - df["入职日期"]).dt.days) / 365.25).round().astype("Int64")
    print(f"基准日 {as_of.date()}：平均年龄 {df['年龄'].mean():.1f}，平均司龄 {df['司龄'].mean():.1f}")

    section("2. 人员结构（部门 / 学历 / 性别）")
    dept = df["部门"].value_counts()
    edu = df["文化程度"].value_counts()
    gender = df["性别"].value_counts()
    print("部门人数：\n", dept.to_string())
    print("\n学历构成：\n", edu.to_string())
    print("\n性别比例：\n", gender.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    dept.head(10).plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_title("各部门人数")
    edu.plot(kind="pie", ax=axes[1], autopct="%1.0f%%", startangle=90)
    axes[1].set_title("学历构成")
    axes[1].set_ylabel("")
    gender.plot(kind="bar", ax=axes[2], color=["#55A868", "#C44E52"])
    axes[2].set_title("性别结构")
    fig.tight_layout()
    print("已保存:", save_fig(fig, "hr_structure.png", OUT))

    section("3. 在职 / 离职分析")
    status = df["员工状态"].value_counts()
    turnover = status.get("离职", 0) / len(df) * 100
    print(status.to_string())
    print(f"累计离职率：{turnover:.1f}%")
    left = df[df["员工状态"] == "离职"]
    if len(left):
        print("\n离职员工平均司龄：", round(left["司龄"].mean(), 1), "年")
        print("离职员工按部门：\n", left["部门"].value_counts().to_string())

    section("4. 职级画像：各部门平均职级与年龄段")
    grade = df.groupby("部门")["职级"].agg(["size", "mean", "max"]).round(2).sort_values("mean", ascending=False)
    print(grade.to_string())
    bins = [0, 25, 30, 35, 40, 50, 100]
    labels = ["25以下", "25-30", "30-35", "35-40", "40-50", "50以上"]
    df["年龄段"] = pd.cut(df["年龄"], bins=bins, labels=labels, right=False)
    age_dist = df["年龄段"].value_counts().sort_index()
    print("\n年龄段分布：\n", age_dist.to_string())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(grade.index, grade["mean"], color="#8172B2")
    ax.set_title("各部门平均职级")
    ax.set_ylabel("平均职级")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    print("已保存:", save_fig(fig, "hr_grade.png", OUT))

    # 输出明细宽表，供复盘
    keep = ["员工工号", "姓名", "性别", "部门", "职称", "职级", "员工状态", "文化程度", "年龄", "司龄"]
    df[keep].to_csv(os.path.join(OUT, "hr_dashboard_detail.csv"), index=False, encoding="utf-8-sig")
    print("\n看板明细已输出到：", OUT)


if __name__ == "__main__":
    main()
