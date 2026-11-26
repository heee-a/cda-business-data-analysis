#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验：维度模型质量检查（第12章 数据模型与数据建模）
=====================================================
【教材依据】第12章配套《阿里云数据中台零售数据模型白皮书》(12.4 MB)，
阐述维度建模的事实表粒度、一致性维度、缓慢变化维、退化维度等规范。
教材第1章「维度模型演示」给出商场销售数据表（销售事实 + 客户/城市/商品维度）。
【Python 迁移要点】把白皮书中的建模规范落成可执行的检查器：
  ①事实表粒度唯一性（业务主键不能重复）
  ②维度表主键唯一性与非空
  ③事实表外键引用完整性（孤立事实）
  ④度量字段数值合法性与缺失
  ⑤维度属性空值率
  ⑥一致性维度检查（企业规模与职工数映射是否冲突）
输出逐项检查报告与问题明细，可直接用于自建数据集。
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "sales_model")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "quality")


def main():
    os.makedirs(OUT, exist_ok=True)
    fact = pd.read_csv(os.path.join(SRC, "sales.csv"), dtype=str)
    dim_cust = pd.read_csv(os.path.join(SRC, "customers.csv"), dtype=str)
    dim_city = pd.read_csv(os.path.join(SRC, "cities.csv"), dtype=str)
    dim_prod = pd.read_csv(os.path.join(SRC, "products.csv"), dtype=str)
    fact["金额"] = pd.to_numeric(fact["金额（$M）"], errors="coerce")

    section("1. 模型概览")
    print(f"事实表 sales：{len(fact):,} 行 × {fact.shape[1]} 列；"
          f"维度表 客户 {len(dim_cust):,}，城市 {len(dim_city)}，商品 {len(dim_prod)}")
    print(f"事实表字段：{list(fact.columns)}")

    results = []

    def check(name, passed, detail):
        results.append({"检查项": name, "结果": "通过" if passed else "不通过", "说明": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}：{detail}")

    section("2. 事实表粒度唯一性")
    grain = ["客户ID", "城市编号", "产品编号"]
    dup = fact.duplicated(grain, keep=False)
    check("事实表粒度唯一", not dup.any(),
          f"业务键 {'+'.join(grain)} 重复 {int(dup.sum())} 行"
          + (f"（示例：{fact.loc[dup, grain].iloc[0].tolist()}）" if dup.any() else ""))

    section("3. 维度表主键唯一性")
    dims = {"客户维度": (dim_cust, "客户ID"), "城市维度": (dim_city, "城市编号"),
            "商品维度": (dim_prod, "产品编号")}
    for name, (df, key) in dims.items():
        d = df.duplicated(key, keep=False)
        nul = df[key].isna().sum() + (df[key].astype(str).str.strip() == "").sum()
        check(f"{name}主键唯一且非空", not d.any() and nul == 0,
              f"{key} 重复 {int(d.sum())} 行，空值 {int(nul)} 个")

    section("4. 外键引用完整性（孤立事实）")
    keys = [("客户ID", dim_cust, "客户ID"), ("城市编号", dim_city, "城市编号"),
            ("产品编号", dim_prod, "产品编号")]
    orphan_detail = []
    for fk, dim, dk in keys:
        valid = set(dim[dk].dropna().astype(str))
        bad = ~fact[fk].astype(str).isin(valid)
        orphan_detail.append((fk, int(bad.sum())))
        check(f"外键 {fk} 引用完整", not bad.any(),
              f"孤立事实 {int(bad.sum())} 行（事实表有、维度表无）")

    section("5. 度量字段合法性")
    nan = fact["金额"].isna().sum()
    neg = (fact["金额"] < 0).sum()
    check("度量「金额($M)」有效", nan == 0 and neg == 0,
          f"缺失 {int(nan)} 行，负值 {int(neg)} 行；"
          f"合计 {fact['金额'].sum():,.1f}，均值 {fact['金额'].mean():.3f}")

    section("6. 维度属性空值率")
    attr_null = []
    for name, (df, _) in dims.items():
        for c in df.columns:
            rate = df[c].astype(str).str.strip().isin(["", "nan", "None"]).mean() * 100
            if rate > 0:
                attr_null.append({"维度表": name, "字段": c, "空值率%": round(rate, 1)})
    if attr_null:
        print(pd.DataFrame(attr_null).to_string(index=False))
    else:
        print("  所有维度属性均无空值")
    check("维度属性完整", len(attr_null) == 0, f"存在空值的属性 {len(attr_null)} 个")

    section("7. 一致性维度：企业规模 ↔ 职工数")
    import re

    def parse_range(s):
        """'300-499人' -> (300,499)；'30000人以上' -> (30000, inf)；'20人以下' -> (0,20)。"""
        s = str(s)
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if not nums:
            return (0, 0)
        if "以上" in s:
            return (nums[0], 10 ** 9)
        if "以下" in s:
            return (0, nums[0])
        if "-" in s and len(nums) >= 2:
            return (nums[0], nums[1])
        return (nums[0], nums[0])

    cust = dim_cust.copy()
    cust["区间"] = cust["企业职工数"].map(parse_range)
    cust["下限"] = cust["区间"].map(lambda x: x[0])
    cust["上限"] = cust["区间"].map(lambda x: x[1])
    mapping = cust.groupby("企业规模").agg(下限=("下限", "min"), 上限=("上限", "max"),
                                        记录数=("企业规模", "size"))
    mapping["职工数取值"] = cust.groupby("企业规模")["企业职工数"].apply(
        lambda s: "、".join(sorted(set(s))))
    print(mapping[["下限", "上限", "记录数", "职工数取值"]].to_string())
    overlaps = []
    sizes = list(mapping.index)
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            a, b = mapping.loc[sizes[i]], mapping.loc[sizes[j]]
            lo = max(a["下限"], b["下限"])
            hi = min(a["上限"], b["上限"])
            if lo <= hi:
                overlaps.append(f"{sizes[i]}∩{sizes[j]}=[{lo},{hi}]人")
    check("企业规模分层无重叠", len(overlaps) == 0,
          f"区间交叉：{'；'.join(overlaps)}" if overlaps else "各规模区间无交叉")

    section("8. 汇总报告")
    report = pd.DataFrame(results)
    print(report.to_string(index=False))
    passed = (report["结果"] == "通过").sum()
    print(f"\n通过 {passed}/{len(report)} 项检查")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#54A24B" if r == "通过" else "#E45756" for r in report["结果"]]
    ax.barh(report["检查项"], [1] * len(report), color=colors)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title(f"维度模型质量检查：通过 {passed}/{len(report)} 项")
    fig.tight_layout()
    save_fig(fig, "quality_check.png", OUT)

    report.to_csv(os.path.join(OUT, "check_report.csv"), index=False, encoding="utf-8-sig")
    if attr_null:
        pd.DataFrame(attr_null).to_csv(
            os.path.join(OUT, "null_attributes.csv"), index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至 {OUT}")


if __name__ == "__main__":
    main()
