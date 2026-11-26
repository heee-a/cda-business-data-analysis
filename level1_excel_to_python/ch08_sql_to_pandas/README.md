# 第8章 SQL 语言基础与 MySQL 入门（SQL → pandas）

**教材素材**：`SQL基础语法.sql`、`1 create_table.sql`、`mysql_csv/*.csv`
**迁移脚本**：`01_sql_fundamentals_in_pandas.py`
**配套数据**：`data/level1/kdd99/*.csv`（GBK 原文件已转 UTF-8 并补表头）

## 覆盖的 9 类 SQL 操作

| SQL 知识点 | pandas 等价写法 |
|---|---|
| SELECT / DISTINCT / WHERE / ORDER BY | 列选择、`drop_duplicates`、布尔过滤、`sort_values` |
| 表达式创建新列、按计算值过滤 | `assign`、`df[df.rate < 0.08]` |
| CASE WHEN 重编码 | `dt.day_of_week` + `map` |
| GROUP BY + HAVING | `groupby().agg()` + 过滤 |
| 笛卡尔积 / 内 / 左 / 右 / 全连接 | `merge(how=...)`、MySQL 不支持的 FULL JOIN 用 pandas 演示 |
| 商业案例三级左连接 | card → disp → clients 链式 `merge` |
| IN 子查询 | `isin()` |
| UNION / UNION ALL | `concat` + `drop_duplicates` |
| CREATE TABLE / VIEW | 生成新 DataFrame 并导出 CSV |

## 与教材的差异

教材的 `KDD99_TRANS` 单文件约 59MB，仓库未收录。分组汇总与日期重编码练习改用同库的 `KDD99_LOANS`（含日期、金额、账户）演示，SQL 语义与 pandas 写法完全一致；连接与子查询练习仍使用原始小表。

运行：`python 01_sql_fundamentals_in_pandas.py`，15 份结果表输出到 `output/`。
