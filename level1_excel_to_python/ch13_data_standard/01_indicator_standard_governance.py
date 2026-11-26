"""第13章 指标体系管理：Excel 数据标准模板 -> Python 治理校验。

教材原素材：第13章 指标体系管理/2、指标数据标准编制模板.xlsx
迁移思路：
    1. 将「标准定义 / 维度定义 / 维度值代码定义」三张表导出为 CSV；
    2. 用 Python 做指标编号唯一性、派生指标口径、单位/类型一致性、维度编码规范性校验；
    3. 输出可复核的质量报告，替代 Excel 手工检查。

运行：python 01_indicator_standard_governance.py
输出：output/indicator_quality_report.csv、output/dimension_quality_report.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "level1" / "ch13_data_standard"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def check_indicators(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    duplicated = set(df.loc[df["编号"].duplicated(keep=False), "编号"])
    for _, r in df.iterrows():
        issues = []
        code = str(r["编号"])
        if not code or code == "nan":
            issues.append("缺少标准编号")
        elif code in duplicated:
            issues.append("标准编号重复")
        if not str(r["中文名称"]).strip():
            issues.append("缺少中文名称")
        if str(r["指标类型"]) == "派生指标" and "/" not in str(r["指标业务口径"]):
            issues.append("派生指标口径缺少计算公式")
        unit, dtype = str(r["计量单位"]), str(r["数据类型"])
        if unit == "%" and dtype not in {"FLOAT", "DECIMAL", "DOUBLE"}:
            issues.append("百分比指标应使用浮点类型")
        if unit.startswith("元") and dtype not in {"INT", "BIGINT", "FLOAT", "DECIMAL"}:
            issues.append("金额指标数据类型不合理")
        if str(r["基本频率"]) not in {"日", "周", "月", "季", "年"}:
            issues.append("更新频率不在允许范围")
        rows.append({
            "编号": code, "中文名称": r["中文名称"], "指标类型": r["指标类型"],
            "计量单位": unit, "数据类型": dtype,
            "校验结果": "通过" if not issues else "不通过",
            "问题说明": "；".join(issues),
        })
    return pd.DataFrame(rows)


def check_dimensions(values: pd.DataFrame, defs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    known = set(defs["维度名称"].astype(str))
    for _, r in values.iterrows():
        issues = []
        name = str(r["维度名称"])
        if name not in known:
            issues.append("维度值未关联到维度定义")
        code = str(r["维度值编码"])
        if not code.isdigit() or len(code) != 2:
            issues.append("维度值编码应为两位数字")
        if not str(r["维度值定义"]).strip():
            issues.append("缺少维度值定义")
        rows.append({
            "维度名称": name, "维度编码": r["维度编码"], "维度值": r["维度值"],
            "维度值编码": code, "校验结果": "通过" if not issues else "不通过",
            "问题说明": "；".join(issues),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ind = pd.read_csv(DATA / "indicator_standard.csv", dtype=str).fillna("")
    dim_def = pd.read_csv(DATA / "dimension_definition.csv", dtype=str).fillna("")
    dim_val = pd.read_csv(DATA / "dimension_values.csv", dtype=str).fillna("")

    iq = check_indicators(ind)
    dq = check_dimensions(dim_val, dim_def)
    iq.to_csv(OUT / "indicator_quality_report.csv", index=False, encoding="utf-8-sig")
    dq.to_csv(OUT / "dimension_quality_report.csv", index=False, encoding="utf-8-sig")

    print(f"指标标准 {len(iq)} 条，通过 {(iq['校验结果'] == '通过').sum()} 条")
    print(iq.to_string(index=False))
    print(f"\n维度值 {len(dq)} 条，通过 {(dq['校验结果'] == '通过').sum()} 条")
    print(dq.to_string(index=False))


if __name__ == "__main__":
    main()
