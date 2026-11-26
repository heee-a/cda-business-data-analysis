#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验：标签体系设计与反欺诈特征治理（第9章 标签体系参考模板）
================================================================
【原 Excel 实现内涵】教材给出两份企业级模板：
  ①「营销类标签.xlsx」——客户关键标签维度/值（60+ 标签、393 个取值）、
    个人客户标签信息表（字段类型/长度/业务含义/取值/取数逻辑）、营销反馈信息表；
  ②「反欺诈特征列表.xlsx」——线上 APP 获客反薅羊毛特征（原始 vs 衍生）。
Excel 里这些信息靠人工翻阅核对，难以发现漏填与口径不一致。
【Python 迁移要点】
  1. 由标签值明细反向构建标签字典（标签→取值数→主数据字段→取数逻辑）；
  2. 校验取数逻辑覆盖率、字段类型/长度合法性、取值项与取数逻辑条数一致性；
  3. 分析反欺诈特征中原始/衍生占比、时间窗口特征命名规律。
"""
import os
import sys
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "tag_system")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "tag_system")


def split_lines(s):
    """单元格里用换行分隔的多值字段 -> 列表。"""
    if not isinstance(s, str):
        return []
    return [x.strip() for x in s.splitlines() if x.strip()]


def main():
    os.makedirs(OUT, exist_ok=True)
    vals = pd.read_csv(os.path.join(SRC, "tag_values.csv"), dtype=str).fillna("")
    fields = pd.read_csv(os.path.join(SRC, "customer_tag_fields.csv"), dtype=str).fillna("")
    feedback = pd.read_csv(os.path.join(SRC, "marketing_feedback.csv"), dtype=str).fillna("")
    fraud = pd.read_csv(os.path.join(SRC, "fraud_features.csv"), dtype=str).fillna("")

    section("1. 标签字典构建（标签→取值）")
    grouped = vals.groupby("标签名称", sort=False).agg(
        取值数=("标签值", "count"),
        取值示例=("标签值", lambda s: " / ".join(list(s)[:5])),
        主数据字段=("主数据字段", lambda s: "、".join(sorted(set(x for x in s if x)))),
        有取数逻辑=("取数逻辑", lambda s: int(s.str.strip().ne("").sum())),
    ).reset_index()
    grouped = grouped.sort_values("取值数", ascending=False)
    print(f"共 {len(grouped)} 个标签，{len(vals)} 个取值，"
          f"平均每个标签 {len(vals) / len(grouped):.1f} 个取值")
    print(grouped.head(15).to_string(index=False))

    section("2. 取数逻辑完整性校验")
    no_logic = vals[vals["取数逻辑"].str.strip() == ""]
    print(f"缺少取数逻辑的标签值：{len(no_logic)} / {len(vals)} "
          f"（{len(no_logic) / len(vals) * 100:.1f}%）")
    dup = vals[vals.duplicated(["标签名称", "标签值"], keep=False)]
    print(f"重复的「标签名+取值」组合：{len(dup)} 条")
    if len(dup):
        print(dup[["标签名称", "标签值"]].head(10).to_string(index=False))

    section("3. 客户标签字段规范校验")
    allowed = {"VARCHAR2", "INT", "DATE", "TIMESTAMP", "NUMBER", "CHAR", "FLOAT"}
    bad_type = fields[~fields["字段类型"].str.upper().isin(allowed) & fields["字段类型"].ne("")]
    bad_len = fields[fields["字段长度"].ne("") & ~fields["字段长度"].str.isdigit()]
    mismatch = []
    for _, r in fields.iterrows():
        n_val = len(split_lines(r["取值"]))
        n_logic = len(split_lines(r["取数逻辑"]))
        if n_val and n_logic and n_val != n_logic:
            mismatch.append((r["字段名称"], n_val, n_logic))
    print(f"字段总数 {len(fields)}；非法类型 {len(bad_type)}；非法长度 {len(bad_len)}；"
          f"取值/逻辑条数不一致 {len(mismatch)}")
    for row in mismatch:
        print(f"  - {row[0]}：{row[1]} 个取值 vs {row[2]} 条逻辑")

    section("4. 营销反馈信息表结构")
    print(feedback[["字段名称", "字段类型", "字段长度", "是否主键", "业务含义"]].to_string(index=False))
    pk = feedback[feedback["是否主键"].str.contains("主键", na=False)]
    print(f"\n主键字段：{', '.join(pk['字段名称']) or '未标注'}")

    section("5. 反欺诈特征分析")
    fraud["衍生"] = fraud["是否衍生"].str.contains("是", na=False)
    cnt = Counter("衍生" if d else "原始" for d in fraud["衍生"])
    print(f"特征总数 {len(fraud)}：原始 {cnt['原始']}，衍生 {cnt['衍生']} "
          f"（衍生占 {cnt['衍生'] / len(fraud) * 100:.1f}%）")
    win = fraud[fraud["字段名"].str.contains(r"_\d+[dhw]?_cnt|_\d+d_", regex=True, na=False)]
    print(f"\n时间窗口计数类特征 {len(win)} 个（形如 *_1d_cnt / *_1h_cnt）：")
    print(win[["字段名", "描述"]].to_string(index=False))

    section("6. 可视化：标签取值数 TOP15")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    top = grouped.head(15).sort_values("取值数")
    axes[0].barh(top["标签名称"], top["取值数"], color="#4C78A8")
    axes[0].set_title("标签取值数 TOP15")
    axes[0].set_xlabel("取值个数")

    axes[1].pie(cnt.values(), labels=list(cnt.keys()), autopct="%1.1f%%",
                colors=["#F58518", "#54A24B"], startangle=90)
    axes[1].set_title("反欺诈特征：原始 vs 衍生")
    fig.tight_layout()
    save_fig(fig, "tag_system_overview.png", OUT)

    grouped.to_csv(os.path.join(OUT, "tag_dictionary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(mismatch, columns=["字段名称", "取值数", "逻辑条数"]).to_csv(
        os.path.join(OUT, "field_mismatch.csv"), index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至 {OUT}")


if __name__ == "__main__":
    main()
