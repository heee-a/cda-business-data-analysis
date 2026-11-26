"""第9章 用户标签体系与用户画像专题：SQL/Excel 画像 -> Python 标签工厂。

教材原素材：第9章/信用卡客户画像.sql、data/*.csv（KDD99 银行零售数据）
迁移思路：
    1. 以客户 client_id 为主键，把账户、持卡、贷款、地域经济、转账流水聚合成一张宽表；
    2. 按教材的「人口属性 / 资产持有 / 风险状态 / 交易行为」四类标签规则打标；
    3. 输出可导入 BI 或推荐系统的客户标签表，并给出标签分布统计。

运行：python 01_customer_profile_tags.py
输出：output/customer_profile_tags.csv、output/tag_distribution.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "level1" / "kdd99"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
AS_OF = pd.Timestamp("1998-12-31")  # 数据集观察截止日


def load() -> dict[str, pd.DataFrame]:
    names = ["kdd99_clients", "kdd99_disp", "kdd99_accounts", "kdd99_card",
             "kdd99_loans", "kdd99_district", "kdd99_order"]
    return {n: pd.read_csv(DATA / f"{n}.csv", dtype=str) for n in names}


def build_wide(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    clients = t["kdd99_clients"].copy()
    clients["birth_date"] = pd.to_datetime(clients["birth_date"], errors="coerce")
    age = ((AS_OF - clients["birth_date"]).dt.days / 365.25).round()
    clients["age"] = age.fillna(-1).astype(int)  # -1 表示出生日期缺失

    # 一个客户可能对应多个账户/卡片，先按客户聚合，避免连接后行数膨胀
    disp = t["kdd99_disp"]
    card = t["kdd99_card"].merge(disp[["disp_id", "client_id"]], on="disp_id", how="left")
    card_feat = (card.groupby("client_id")
                 .agg(card_count=("card_id", "nunique"),
                      card_types=("type", lambda s: "、".join(sorted(set(s.dropna())))))
                 .reset_index())

    acc_feat = (disp.groupby("client_id")
                .agg(account_count=("account_id", "nunique")).reset_index())

    loans = t["kdd99_loans"].copy()
    loans["amount_num"] = pd.to_numeric(loans["amount"], errors="coerce")
    loan_feat = (loans.merge(disp[["client_id", "account_id"]], on="account_id", how="left")
                 .groupby("client_id")
                 .agg(loan_count=("loan_id", "nunique"),
                      loan_amount_sum=("amount_num", "sum"),
                      loan_bad_count=("status", lambda s: int((s == "B").sum())))
                 .reset_index())

    order = t["kdd99_order"].copy()
    order["amount_num"] = pd.to_numeric(order["amount"], errors="coerce")
    order_feat = (order.merge(disp[["client_id", "account_id"]], on="account_id", how="left")
                  .groupby("client_id")
                  .agg(order_count=("order_id", "nunique"),
                       order_amount_sum=("amount_num", "sum"))
                  .reset_index())

    dist = t["kdd99_district"].copy()
    dist["GDP"] = pd.to_numeric(dist["GDP"], errors="coerce")
    dist["gdp_tier"] = pd.qcut(dist["GDP"], 3, labels=["低", "中", "高"]).astype(str)
    dist = dist[["A1", "gdp_tier"]].rename(columns={"A1": "district_id"})

    wide = (clients[["client_id", "sex", "age", "district_id"]]
            .merge(dist, on="district_id", how="left")
            .merge(acc_feat, on="client_id", how="left")
            .merge(card_feat, on="client_id", how="left")
            .merge(loan_feat, on="client_id", how="left")
            .merge(order_feat, on="client_id", how="left"))
    for c in ["card_count", "account_count", "loan_count", "loan_amount_sum",
              "loan_bad_count", "order_count", "order_amount_sum"]:
        wide[c] = wide[c].fillna(0)
    return wide


def add_tags(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    df["性别标签"] = df["sex"].map({"男": "男性", "女": "女性"}).fillna("未知")
    df["年龄段标签"] = pd.cut(df["age"], [0, 30, 45, 60, 200],
                           labels=["30岁以下", "30-45岁", "45-60岁", "60岁以上"]).astype(str)
    df["地域经济标签"] = df["gdp_tier"].fillna("未知") + "GDP地区"
    df["持卡标签"] = np.where(df["card_count"] > 0, "持卡客户", "未持卡客户")
    df["贷款状态标签"] = np.where(df["loan_bad_count"] > 0, "有关注贷款",
                              np.where(df["loan_count"] > 0, "正常贷款客户", "无贷款客户"))
    df["贷款金额标签"] = pd.cut(df["loan_amount_sum"], [-1, 0, 50000, 150000, np.inf],
                            labels=["无贷款", "小额贷款", "中额贷款", "大额贷款"]).astype(str)
    df["交易活跃度标签"] = pd.cut(df["order_amount_sum"], [-1, 0, 5000, 50000, np.inf],
                              labels=["无交易", "低频", "中频", "高频"]).astype(str)

    # 客户价值综合分：交易金额 + 贷款金额，各取对数后加权，按三分位分层
    score = np.log1p(df["order_amount_sum"]) + 0.5 * np.log1p(df["loan_amount_sum"])
    df["价值得分"] = score.round(3)
    df["客户价值标签"] = pd.qcut(score, 3, labels=["低价值", "中价值", "高价值"]).astype(str)
    return df


def main() -> None:
    tags = add_tags(build_wide(load()))
    cols = ["client_id", "性别标签", "年龄段标签", "地域经济标签", "持卡标签",
            "贷款状态标签", "贷款金额标签", "交易活跃度标签", "客户价值标签",
            "价值得分", "account_count", "card_count", "loan_count", "order_count"]
    tags[cols].to_csv(OUT / "customer_profile_tags.csv", index=False, encoding="utf-8-sig")

    tag_cols = [c for c in cols if c.endswith("标签")]
    summary = pd.concat([
        tags[c].value_counts().rename_axis("标签值").rename("客户数").reset_index().assign(标签体系=c)
        for c in tag_cols
    ], ignore_index=True)[["标签体系", "标签值", "客户数"]]
    summary.to_csv(OUT / "tag_distribution.csv", index=False, encoding="utf-8-sig")

    print(f"生成客户标签 {len(tags)} 行，标签体系 {len(tag_cols)} 类")
    print("\n客户价值分布：")
    print(summary[summary["标签体系"] == "客户价值标签"].to_string(index=False))
    print("\n贷款状态分布：")
    print(summary[summary["标签体系"] == "贷款状态标签"].to_string(index=False))


if __name__ == "__main__":
    main()
