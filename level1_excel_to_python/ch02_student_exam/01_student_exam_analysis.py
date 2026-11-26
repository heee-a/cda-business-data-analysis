#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章 PowerBI 函数练习 —— 学生考试案例分析（Python 迁移）
==========================================================
教材用 Power BI「学生案例.pbix」把学生信息、教师信息、考试记录三张表关联，
再用 DAX 求各班平均分、学科对比与成绩排名。本脚本用 pandas 完整复刻：

1. 数据关联：学生表 + 教师表（班级→班主任）+ 考试记录（学号→三科成绩）；
2. 成绩统计：语文/数学/英语的描述统计、总分与平均分；
3. 班级对比：各班各科平均分、及格率；
4. 性别对比：男女生各科平均分差异；
5. 排名：按总分排名并标记前三名；
6. 可视化：班级平均分对比、学科平均分、总分分布。

运行：python 01_student_exam_analysis.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import matplotlib.pyplot as plt
from common import section, save_fig, excel_serial_to_datetime, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "exam")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    os.makedirs(OUT, exist_ok=True)
    students = pd.read_csv(os.path.join(SRC, "students.csv"), dtype=str)
    teachers = pd.read_csv(os.path.join(SRC, "teachers.csv"), dtype=str)
    exam = pd.read_csv(os.path.join(SRC, "records.csv"), dtype=str)
    students.columns = ["学号", "姓名", "性别", "班级", "出生日期"]
    teachers.columns = ["班级", "班主任姓名"]
    exam.columns = ["学号", "语文", "数学", "英语"]

    section("1. 三表关联：学生 + 班主任 + 成绩")
    df = students.merge(teachers, on="班级", how="left").merge(exam, on="学号", how="left")
    for c in ["语文", "数学", "英语"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["班级"] = pd.to_numeric(df["班级"], errors="coerce")
    df["出生日期"] = excel_serial_to_datetime(df["出生日期"])
    df["总分"] = df[["语文", "数学", "英语"]].sum(axis=1)
    df["平均分"] = df[["语文", "数学", "英语"]].mean(axis=1).round(2)
    print(f"关联后 {len(df)} 名学生；字段：{list(df.columns)}")
    print(df[["学号", "姓名", "班级", "班主任姓名", "语文", "数学", "英语", "总分"]].head(6).to_string(index=False))

    section("2. 学科描述统计")
    stat = df[["语文", "数学", "英语"]].agg(["mean", "std", "min", "max"]).round(2)
    print(stat.to_string())

    section("3. 班级对比：各科平均分与及格率")
    by_class = df.groupby("班级").agg(
        人数=("学号", "size"),
        语文均分=("语文", "mean"), 数学均分=("数学", "mean"), 英语均分=("英语", "mean"),
        总均分=("平均分", "mean")
    ).round(2)
    for c in ["语文", "数学", "英语"]:
        by_class[c + "及格率"] = (df.assign(通过=df[c] >= 60).groupby("班级")["通过"].mean() * 100).round(1)
    print(by_class.to_string())
    by_class.to_csv(os.path.join(OUT, "class_summary.csv"), encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    by_class[["语文均分", "数学均分", "英语均分"]].plot(kind="bar", ax=ax)
    ax.set_title("各班学科平均分对比")
    ax.set_ylabel("平均分")
    ax.set_xlabel("班级")
    fig.tight_layout()
    print("已保存:", save_fig(fig, "class_compare.png", OUT))

    section("4. 性别对比")
    gender_map = {"1": "男", "2": "女"}
    df["性别"] = df["性别"].map(gender_map).fillna(df["性别"])
    by_gender = df.groupby("性别")[["语文", "数学", "英语", "平均分"]].mean().round(2)
    print(by_gender.to_string())

    section("5. 总分排名（RANK 等价实现）")
    df["排名"] = df["总分"].rank(method="min", ascending=False).astype(int)
    top = df.sort_values("总分", ascending=False).head(10)
    print(top[["排名", "姓名", "班级", "总分", "平均分"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df["总分"].dropna(), bins=15, color="#4C72B0", edgecolor="white")
    ax.set_title("学生总分分布")
    ax.set_xlabel("总分")
    ax.set_ylabel("人数")
    fig.tight_layout()
    print("已保存:", save_fig(fig, "score_distribution.png", OUT))

    cols = ["学号", "姓名", "性别", "班级", "班主任姓名", "语文", "数学", "英语", "总分", "平均分", "排名"]
    df[cols].to_csv(os.path.join(OUT, "student_exam_wide.csv"), index=False, encoding="utf-8-sig")
    print("\n学生成绩宽表已输出到：", OUT)


if __name__ == "__main__":
    main()
