#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 13（第1章）：关系/维度数据模型演示（对应「用 Power BI 演示数据模型的用处」）
==================================================================
【原 Excel/Power BI 实现内涵】
    教材用一张“银行零售”关系模型（district / clients / disp / accounts /
    loans / trans 等互相通过主键-外键关联）演示：有了数据模型，
    才能把“事实表(交易/贷款)”与“维度表(地区/客户)”关联起来做跨表分析。
    Power BI 里是拖拽建立关系；Excel 里是 VLOOKUP/Power Query 合并。

【Python 迁移要点】
    用 pandas 读入各张表，按主键-外键做 merge（等价于 Power BI 的关系），
    再完成两类跨表分析：
      (1) 各地区(district 维度) 的贷款金额分布；
      (2) 各地区的总交易金额（事实表 trans 聚合后关联维度）。
    这正是“数据模型的用处”：把分散在多张表里的信息打通。
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_LEVEL1, save_fig, section

SRC = os.path.join(DATA_LEVEL1, "datamodel")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "datamodel")


def main():
    section("1. 载入各张表（维度表 + 事实表）")
    # 该数据集为捷克银行公开数据，文本字段采用 cp1250 编码（utf-8 会解码失败）
    district = pd.read_csv(os.path.join(SRC, "district.csv"), encoding="cp1250")
    district = district.rename(columns={"A1": "district_id"})
    accounts = pd.read_csv(os.path.join(SRC, "accounts.csv"), encoding="cp1250")[["account_id", "district_id"]]
    clients = pd.read_csv(os.path.join(SRC, "clients.csv"), encoding="cp1250")[["client_id", "district_id"]]
    disp = pd.read_csv(os.path.join(SRC, "disp.csv"), encoding="cp1250")
    loans = pd.read_csv(os.path.join(SRC, "loans.csv"), encoding="cp1250")[["account_id", "amount"]]
    loans = loans.rename(columns={"amount": "loan_amount"})
    trans = pd.read_csv(os.path.join(SRC, "trans.csv"), encoding="cp1250",
                        low_memory=False, usecols=["account_id", "amount"])
    trans = trans.rename(columns={"amount": "trans_amount"})
    # 交易金额原始为文本，统一转数值（无法解析的置 NaN 后丢弃）
    trans["trans_amount"] = pd.to_numeric(trans["trans_amount"], errors="coerce")
    print(f"账户 {len(accounts):,}，客户 {len(clients):,}，贷款 {len(loans):,}，"
          f"交易 {len(trans):,}，地区 {len(district)}")

    section("2. 建立关系（merge）——等价于 Power BI 里拖出的关系线")
    # 账户 -> 地区（维度）
    acc_dist = accounts.merge(district[["district_id"]], on="district_id", how="left")
    # 贷款 -> 账户 -> 地区
    loan_geo = loans.merge(acc_dist, on="account_id", how="left")
    # 客户 -> 账户（谁拥有账户，仅保留 OWNER）
    owner = disp[disp["type"].astype(str).str.upper() == "OWNER"][["client_id", "account_id"]]
    client_geo = owner.merge(acc_dist, on="account_id", how="left")

    section("3. 跨表分析(1)：各地区贷款总额与笔数")
    by_region = (loan_geo.dropna(subset=["district_id"])
                        .groupby("district_id")
                        .agg(贷款笔数=("loan_amount", "count"),
                             贷款总额=("loan_amount", "sum"),
                             平均贷款=("loan_amount", "mean"))
                        .sort_values("贷款总额", ascending=False))
    by_region.index = by_region.index.astype(int).astype(str)
    print(by_region.head(10))

    section("4. 跨表分析(2)：各地区总交易金额")
    trans_geo = trans.merge(acc_dist, on="account_id", how="left")
    trans_by_region = (trans_geo.dropna(subset=["district_id"])
                       .groupby("district_id")["trans_amount"].sum()
                       .sort_values(ascending=False))
    print(trans_by_region.head(5))

    section("5. 可视化：Top 10 地区贷款总额")
    top = by_region.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top.index.astype(str), top["贷款总额"] / 1e6, color="#4C72B0")
    ax.set_title("各地区贷款总额 Top10 (百万)")
    ax.set_ylabel("贷款总额 (百万)"); plt.xticks(rotation=45, ha="right")
    p = save_fig(fig, "loan_by_region.png", OUT)
    print("已保存：", p)
    print("\n核心结论：单看 loans 表只能得到“每笔贷款金额”；引入 district 维度后，"
          "才能回答“哪个地区的贷款需求最旺盛”——这就是数据模型打通多张表的价值。")


if __name__ == "__main__":
    main()
