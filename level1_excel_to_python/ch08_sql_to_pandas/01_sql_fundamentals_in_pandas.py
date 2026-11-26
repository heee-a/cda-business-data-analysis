"""第8章 SQL 语言基础与 MySQL 入门：SQL -> pandas 等价实现。

教材原素材：第8章/SQL基础语法.sql、1 create_table.sql、mysql_csv/*.csv
迁移思路：不依赖 MySQL 服务，用 pandas 复现教程中的 9 类 SQL 操作，
         重点理解「选择 / 过滤 / 重编码 / 分组 / 连接 / 子查询 / 集合操作 / 视图」的本质。

说明：教材的 KDD99_TRANS 单文件约 59MB，仓库未收录；分组汇总练习改用
      同结构的 KDD99_ORDER（交易流水）演示，分析逻辑完全一致。

运行：python 01_sql_fundamentals_in_pandas.py
输出：output/sql01_select.csv ... sql09_view.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "level1" / "kdd99"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, dtype=str)


def save(df: pd.DataFrame, name: str, title: str) -> None:
    print(f"\n--- {title} ({len(df)} 行) ---")
    print(df.head(8).to_string(index=False))
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def main() -> None:
    sale = read("sale.csv")
    for c in ("year", "sale", "profit"):
        sale[c] = pd.to_numeric(sale[c])
    accounts = read("kdd99_accounts.csv")
    card = read("kdd99_card.csv")
    clients = read("kdd99_clients.csv")
    disp = read("kdd99_disp.csv")
    loans = read("kdd99_loans.csv")
    loans["amount_num"] = pd.to_numeric(loans["amount"], errors="coerce")
    order = read("kdd99_order.csv")
    order["amount_num"] = pd.to_numeric(order["amount"], errors="coerce")
    loans["date"] = pd.to_datetime(loans["date"], errors="coerce")
    one, two = read("one.csv"), read("two.csv")
    t1, t2 = read("table1.csv"), read("table2.csv")

    # SQL1/2 SELECT、DISTINCT、WHERE、ORDER BY
    save(sale.loc[sale["year"] == 2010, ["year", "market", "sale", "profit"]]
         .sort_values("year"), "sql01_select.csv", "SQL1 选择列 + 过滤 + 排序")
    save(sale[["year"]].drop_duplicates(), "sql02_distinct.csv", "SQL2 DISTINCT")

    # SQL3 表达式创建新列 + 用计算值过滤
    sale2 = sale.assign(rate=sale["profit"] / sale["sale"])
    save(sale2.loc[sale2["rate"] < 0.08], "sql03_expression.csv", "SQL3 计算列 + 条件过滤")

    # SQL4 CASE 重编码：贷款日期 -> 工作日 / 周末（教材用 TRANS.date，此处用 LOANS.date）
    loans2 = loans.copy()
    loans2["weekday"] = loans2["date"].dt.day_of_week  # 周一=0，与 MySQL WEEKDAY 一致
    loans2["weekday_type"] = loans2["weekday"].map(
        lambda d: "周末" if d >= 5 else "工作日"
    )
    save(loans2[["loan_id", "account_id", "date", "amount_num", "weekday", "weekday_type"]],
         "sql04_case.csv", "SQL4 CASE WHEN 重编码")

    # SQL5 GROUP BY + HAVING：按账户统计贷款笔数与金额，筛出多笔贷款账户
    grouped = (loans2.groupby("account_id")
               .agg(loan_count=("loan_id", "size"),
                    mean_amount=("amount_num", "mean"), sum_amount=("amount_num", "sum"))
               .reset_index())
    grouped = grouped[grouped["loan_count"] > 1]
    save(grouped, "sql05_group_having.csv", "SQL5 GROUP BY + HAVING")

    # SQL6 连接：笛卡尔积 / 内连接 / 左连接 / 右连接 / 全连接(MySQL 不支持，pandas 可模拟)
    cross = t1.merge(t2, how="cross", suffixes=("_1", "_2"))
    save(cross, "sql06_cross_join.csv", "SQL6 笛卡尔积")
    save(t1.merge(t2, on="id"), "sql07_inner_join.csv", "SQL6 内连接")
    save(t1.merge(t2, on="id", how="left"), "sql08_left_join.csv", "SQL6 左连接")
    save(t1.merge(t2, on="id", how="right"), "sql09_right_join.csv", "SQL6 右连接")
    save(t1.merge(t2, on="id", how="outer"), "sql10_full_join.csv", "SQL6 全连接(pandas)")

    # 商业问题：信用卡 -> 账户关系 -> 客户信息 的三级左连接
    card_join = (card.merge(disp, on="disp_id", how="left")
                 .merge(clients, on="client_id", how="left")
                 [["card_id", "disp_id", "issued", "type_x", "sex", "birth_date", "district_id"]]
                 .rename(columns={"type_x": "card_type"}))
    save(card_join, "sql11_business_join.csv", "SQL6 商业案例：三级左连接")

    # SQL7 子查询：违约贷款账户的交易流水（此处用 order 代替 trans）
    bad_accounts = loans.loc[loans["status"] == "B", "account_id"]
    sub = order[order["account_id"].isin(bad_accounts)]
    save(sub, "sql12_subquery.csv", "SQL7 IN 子查询")

    # SQL8 集合操作 UNION / UNION ALL
    save(pd.concat([one, two]).drop_duplicates(), "sql13_union.csv", "SQL8 UNION DISTINCT")
    save(pd.concat([one, two], ignore_index=True), "sql14_union_all.csv", "SQL8 UNION ALL")

    # SQL9 建表 / 视图：Python 中即「生成新的 DataFrame并导出」
    view = sale.loc[sale["year"].isin([2010, 2011])].copy()
    save(view, "sql15_view.csv", "SQL9 CREATE TABLE / VIEW")

    print("\n全部 SQL 练习已用 pandas 复现，结果见 output/ 目录。")


if __name__ == "__main__":
    main()
