"""第7章 指标体系构建方法：Excel 模板 -> Python 指标体系建设。

教材原素材：第7章 指标体系构建方法/1、指标体系设计模板.xlsx（Step1/Step2）
迁移思路：
    1. 把 Step2 的「指标名称/计算规则/口径/维度/指标类型/关联指标」读成结构化表；
    2. 用 Python 校验指标层级、类型与上游依赖，形成可维护的指标字典；
    3. 用 KDD99 银行零售数据给模板中的核心指标算出示例值，验证口径是否可落地。

运行：python 01_metric_system_design.py
输出：output/metric_dictionary.csv、output/metric_dependency_check.csv、output/metric_demo_values.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "level1"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def load_template() -> pd.DataFrame:
    df = pd.read_csv(DATA / "ch07_metric_system" / "metric_system_template.csv", dtype=str)
    df = df.fillna("")
    # Excel 中「指标层级」只在每个分组首行填写，这里向下填充
    df["指标层级"] = df["指标层级"].replace("", pd.NA).ffill().fillna("未分组")
    df["指标名称（KPI）"] = df["指标名称（KPI）"].str.strip()
    df = df[df["指标名称（KPI）"] != ""].reset_index(drop=True)
    return df


def parse_dependencies(formula: str, known_names: set[str], current: str = "") -> list[str]:
    """从计算规则中识别上游指标名（优先用「关联指标」字段，其次从公式文本匹配）。"""
    hits = [name for name in known_names if name and name in formula and name != current]
    # 去掉被更长名称包含的短名称，避免「放款用户数」与「放款转化率」互相误判
    return [h for h in hits if not any(h != other and h in other for other in hits)]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    names = set(df["指标名称（KPI）"])
    rows = []
    for _, r in df.iterrows():
        name = r["指标名称（KPI）"]
        kind = r["指标类型"]
        formula = r["计算规则"]
        related = [x.strip() for x in re.split(r"[,，/、]", r["关联指标"]) if x.strip()]
        deps = sorted(set(related + parse_dependencies(formula, names, name)))
        issues = []
        if not formula:
            issues.append("缺少计算规则")
        if kind not in {"基础指标", "派生指标"}:
            issues.append(f"指标类型异常：{kind}")
        if kind == "派生指标" and not deps:
            issues.append("派生指标缺少上游依赖")
        for dep in deps:
            if dep not in names:
                issues.append(f"上游指标『{dep}』未在指标字典中定义")
        rows.append({
            "指标层级": r["指标层级"], "指标名称": name, "指标类型": kind,
            "计量值": r["值类型"], "维度": r["维度"], "计算规则": formula,
            "上游依赖": "、".join(deps), "校验结果": "通过" if not issues else "不通过",
            "问题说明": "；".join(issues),
        })
    return pd.DataFrame(rows)


def demo_values(names: set[str]) -> pd.DataFrame:
    """用教材 KDD99 数据给可计算的指标生成示例值（口径假设见 README）。"""
    clients = pd.read_csv(DATA / "kdd99" / "kdd99_clients.csv", dtype=str)
    loans = pd.read_csv(DATA / "kdd99" / "kdd99_loans.csv", dtype=str)
    loans["amount_num"] = pd.to_numeric(loans["amount"], errors="coerce")

    raw = {
        "注册用户数": len(clients),
        "进件用户数": len(loans),
        "审核通过用户数": int(loans["status"].isin(["A", "C"]).sum()),
        "放款用户数": int((loans["status"] == "A").sum()),
        "放款金额": float(loans.loc[loans["status"] == "A", "amount_num"].sum()),
    }
    derived = {
        "审核通过率": raw["审核通过用户数"] / raw["进件用户数"],
        "放款转化率": raw["放款用户数"] / max(raw["审核通过用户数"], 1),
        "放款件均": raw["放款金额"] / max(raw["放款用户数"], 1),
    }
    rows = [{"指标名称": k, "示例值": v, "取值来源": "KDD99 clients/loans"}
            for k, v in {**raw, **derived}.items() if k in names]
    return pd.DataFrame(rows)


def main() -> None:
    tpl = load_template()
    check = validate(tpl)
    values = demo_values(set(tpl["指标名称（KPI）"]))

    check.to_csv(OUT / "metric_dependency_check.csv", index=False, encoding="utf-8-sig")
    tpl.to_csv(OUT / "metric_dictionary.csv", index=False, encoding="utf-8-sig")
    values.to_csv(OUT / "metric_demo_values.csv", index=False, encoding="utf-8-sig")

    print(f"指标总数：{len(check)}，校验通过：{(check['校验结果'] == '通过').sum()}")
    print("\n依赖关系：")
    print(check[["指标名称", "指标类型", "上游依赖", "校验结果"]].to_string(index=False))
    print("\n示例指标值：")
    print(values.to_string(index=False))


if __name__ == "__main__":
    main()
