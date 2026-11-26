#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章 PowerBI 基本操作 —— ETL 数据整理（Power Query → pandas）
=============================================================
教材在 Power BI 的 Power Query 里演示了 4 类高频数据整理操作。本脚本用
pandas 逐一复刻，并说明每一步对应的 Power Query 操作：

1. 按分隔符拆分列（拆分到行）：一个订单号单元格里塞了多个订单号，按逗号拆成多列再逆透视到行；
2. 脏数据清洗：把 "#N/A" 占位符、空值、金额=0 的无效行标记/剔除，日期序列号转日期；
3. 横向合并（合并查询 / VLOOKUP）：学生表、课程表、选课表三表按主键合并成宽表；
4. 纵向合并（追加查询）：字段一致与字段不一致两种追加，演示列名对齐策略。

运行：python 01_powerquery_etl.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from common import section, excel_serial_to_datetime, DATA_LEVEL1

SRC = os.path.join(DATA_LEVEL1, "etl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    os.makedirs(OUT, exist_ok=True)

    section("1. 拆分列：按逗号拆开订单号并展开到行")
    raw = pd.read_csv(os.path.join(SRC, "split_column.csv"), dtype=str)
    print("原始数据：", raw.shape)
    print(raw.head().to_string(index=False))
    # Power Query「按分隔符拆分列→拆分为行」在 pandas 里用 str.split + explode
    raw["订单号"] = raw["订单号"].str.split(",")
    exploded = raw.explode("订单号").reset_index(drop=True)
    exploded["订单号"] = exploded["订单号"].str.strip()
    print(f"\n拆分前 {len(raw)} 行 → 拆分后 {len(exploded)} 行")
    print(exploded[["订单号", "收货地址", "数量"]].to_string(index=False))
    exploded.to_csv(os.path.join(OUT, "etl01_split_column.csv"), index=False, encoding="utf-8-sig")

    section("2. 数据清洗：#N/A 占位、空值、无效金额与日期序列号")
    dirty = pd.read_csv(os.path.join(SRC, "cleaning.csv"), dtype=str)
    print("清洗前行数：", len(dirty))
    print("各列缺失情况：")
    print(dirty.isna().sum().to_string())
    clean = dirty.copy()
    # 2.1 把文本占位符统一成 NaN（Power Query「替换值」）
    clean = clean.replace(["#N/A", "N/A", "null", "", "未知"], pd.NA)
    # 2.2 数值列强制转数值，无法解析的置 NaN
    for c in ["实付金额", "下单数量"]:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    # 2.3 剔除金额为 0 或缺失的无效行（退货/异常单）
    invalid = clean[clean["实付金额"].isna() | (clean["实付金额"] <= 0)]
    print(f"\n识别到 {len(invalid)} 条无效记录（金额缺失或为 0）")
    print(invalid[["订单号", "实付金额", "下单数量"]].to_string(index=False))
    clean = clean.dropna(subset=["实付金额"])
    clean = clean[clean["实付金额"] > 0].reset_index(drop=True)
    # 2.4 Excel 日期序列号转真实日期（origin 1899-12-30）
    clean["下单日期"] = excel_serial_to_datetime(clean["order_date"])
    print(f"\n清洗后 {len(clean)} 行；日期范围：{clean['下单日期'].min().date()} ~ {clean['下单日期'].max().date()}")
    print(f"有效成交额合计：{clean['实付金额'].sum():,.2f}")
    clean.to_csv(os.path.join(OUT, "etl02_cleaned.csv"), index=False, encoding="utf-8-sig")

    section("3. 横向合并：学生 × 课程 × 选课（合并查询）")
    course = pd.read_csv(os.path.join(SRC, "course.csv"), dtype=str)
    student = pd.read_csv(os.path.join(SRC, "student.csv"), dtype=str)
    score = pd.read_csv(os.path.join(SRC, "score.csv"), dtype=str)
    course.columns = ["课程号", "课程名", "学分"]
    student.columns = ["学号", "姓名", "性别", "出生日期", "系别", "手机号"]
    score.columns = ["学号", "课程号", "成绩"]
    # 先选课表分别左连接学生表、课程表，等价于 Power Query 的两次「合并查询」
    merged = (score.merge(student, on="学号", how="left")
                   .merge(course, on="课程号", how="left"))
    merged["成绩"] = pd.to_numeric(merged["成绩"], errors="coerce")
    merged["学分"] = pd.to_numeric(merged["学分"], errors="coerce")
    merged["出生日期"] = excel_serial_to_datetime(merged["出生日期"]).dt.date
    print(f"合并后宽表：{merged.shape}")
    print(merged[["学号", "姓名", "系别", "课程名", "成绩", "学分"]].head(8).to_string(index=False))
    # 类似 DAX 的汇总：按系别统计平均成绩
    by_dept = merged.groupby("系别").agg(选课人次=("成绩", "size"), 平均成绩=("成绩", "mean"),
                                        修读学分=("学分", "sum")).round(2)
    print("\n按系别汇总：")
    print(by_dept.to_string())
    merged.to_csv(os.path.join(OUT, "etl03_merged.csv"), index=False, encoding="utf-8-sig")

    section("4. 纵向合并：追加查询（字段名一致 / 不一致）")
    q1 = pd.read_csv(os.path.join(SRC, "append_same_q1.csv"))
    q2 = pd.read_csv(os.path.join(SRC, "append_same_q2.csv"))
    append_same = pd.concat([q1, q2], ignore_index=True)
    print("字段一致时直接追加，行数：", len(append_same))
    print(append_same.to_string(index=False))
    d1 = pd.read_csv(os.path.join(SRC, "append_diff_q1.csv"))
    d2 = pd.read_csv(os.path.join(SRC, "append_diff_q2.csv"))
    print("\n字段不一致：第一季度第三列为「销售额」，第二季度为「金额」")
    # 对齐列名后再追加（Power Query 会按列名对齐，缺失列填 null）
    d2 = d2.rename(columns={"金额": "销售额"})
    append_diff = pd.concat([d1, d2], ignore_index=True)
    append_diff["季度来源"] = ["第一季度"] * len(d1) + ["第二季度"] * len(d2)
    print(append_diff.to_string(index=False))
    print(f"\n两季度销售额合计：{pd.to_numeric(append_diff['销售额']).sum():,.0f}")
    append_same.to_csv(os.path.join(OUT, "etl04_append_same.csv"), index=False, encoding="utf-8-sig")
    append_diff.to_csv(os.path.join(OUT, "etl05_append_diff.csv"), index=False, encoding="utf-8-sig")

    print("\n全部 ETL 结果已输出到：", OUT)


if __name__ == "__main__":
    main()
