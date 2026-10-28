# CDA《商业数据分析》Python 实验仓库

![License](https://img.shields.io/github/license/heee-a/cda-business-data-analysis)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![CDA](https://img.shields.io/badge/CDA-Level%201--4-2ea44f)
![Repo Size](https://img.shields.io/github/repo-size/heee-a/cda-business-data-analysis)

> 开源地址：GitHub https://github.com/heee-a/cda-business-data-analysis ｜ Gitee 同步中

本仓库是 **CDA（Certified Data Analyst）数据分析师认证教材《商业数据分析》** 全部实验内容的 Python 复现与迁移工程。

教材原始配套实验分别用 **Excel / Power BI** 和 **Jupyter Notebook（Python）** 实现。本项目按教材要求完成两件事：

1. **Excel / Power BI 实验 → Python 迁移**：理解原实验的分析内涵，用 `pandas / numpy / scipy / statsmodels / scikit-learn / matplotlib / seaborn` 重新实现，做到“换工具、不丢分析逻辑”。
2. **Jupyter Notebook 实验 → 纯 `.py` 文件**：把教材自带的所有 `.ipynb` 转化为可直接运行的 Python 脚本，并梳理代码本质。

> 环境说明：本机 Python 依赖通过离线 wheel 安装（见 `tools/download_wheels.py`）。运行脚本前请先 `pip install -r requirements.txt`。

---

## 目录结构

```
cda-business-data-analysis/
├── level1_excel_to_python/        # 一级教材：Excel/PowerBI 实验的 Python 迁移（全新重写）
│   ├── common.py                  # 公共工具：中文字体、路径、Excel 日期转换、存图
│   ├── ch01_data_model/           # 第1章 关系/维度数据模型（银行零售关系模型演示）
│   ├── ch02_analysis_paradigms/   # 第2章 11 种分析范式（RFM、波士顿矩阵、漏斗…）
│   └── ch10_time_series_decomposition/  # 第10章 时间序列效应分解
├── level2_quant_strategy_python/  # 二级教材：量化策略分析（由 .ipynb 转换）
├── level3_agile_mining_python/    # 三级教材：敏捷数据挖掘（由 .ipynb 转换）
├── level4_finance_cases_python/   # 四级教材：金融案例实战（由 .ipynb 转换）
├── data/                          # 各实验所用数据集（含 LFS 大文件）
├── tools/                         # 构建工具：notebook→py 转换、离线 wheel 下载等
├── requirements.txt
├── LICENSE
└── .gitattributes                 # 大文件走 Git LFS
```

---

## Level 1：Excel / Power BI → Python 迁移（重点）

这部分不是简单的“翻译”，而是**理解实验内涵后用 Python 重做**，共 13 个脚本，均已在本地验证可运行并产出图表。

### 第2章 · 11 种数据分析范式（`ch02_analysis_paradigms/`）

| 脚本 | 对应范式 | 说明 |
|------|----------|------|
| `01_rfm_model.py` | RFM 模型 | 按 Recency / Frequency / Monetary 给客户分 8 段 |
| `02_boston_matrix.py` | 波士顿(BCG)矩阵 | 增长率 vs 相对市场份额四象限 |
| `03_user_loyalty.py` | 用户忠诚度 | 生命周期(潜在/易流失/活跃)+F/M 评分→忠诚/普通/低价值 |
| `04_cohort_analysis.py` | 同期群分析 | 留存率热力图 |
| `05_funnel.py` | 漏斗分析 | 各阶段人数与转化率 |
| `06_correlation.py` | 相关分析 | Pearson / Spearman |
| `07_trend_analysis.py` | 趋势分析 | 趋势线 + 三期移动平均 |
| `08_comparison_analysis.py` | 对比分析 | 分组柱状对比 |
| `09_composition_analysis.py` | 构成分析 | 饼图 + 百分比柱图 |
| `10_distribution_analysis.py` | 分布分析 | 直方图 + 箱线图 |
| `11_relationship_analysis.py` | 关系分析 | 分组柱 + 散点 |

### 第1章 · 数据模型（`ch01_data_model/`）

`01_relational_model.py`：用捷克银行零售公开数据（district / clients / disp / accounts / loans / trans 六张表）演示**关系/维度模型**的价值——
通过主键-外键 `merge`（等价于 Power BI 里拖出的关系线），把“事实表(交易/贷款)”与“维度表(地区)”打通，回答“哪个地区贷款需求最旺盛”这类跨表问题。

### 第10章 · 时间序列效应分解（`ch10_time_series_decomposition/`）

`01_effect_decomposition.py`：对航空客流、信用卡消费额做 `statsmodels` 乘法/加法分解（趋势+季节+不规则），并手工计算季节指数。

> 运行方式（以第2章 RFM 为例）：
> ```bash
> cd level1_excel_to_python/ch02_analysis_paradigms
> python 01_rfm_model.py
> ```
> 图表默认保存到各脚本下的 `output/` 目录。

---

## Level 2 / 3 / 4：Jupyter Notebook → `.py`（由工具自动转换）

`level2_quant_strategy_python/`、`level3_agile_mining_python/`、`level4_finance_cases_python/` 下共 **58 个** `.py` 文件，由教材配套的 `.ipynb` 经 `tools/nb_to_py.py` 转换而来：

- Markdown 单元格 → `#` 注释，保留全部讲解；
- 代码单元格 → 原样保留（magics 行自动注释）；
- 文件名沿用原 notebook 标题，便于对照教材。

每个级别目录下均有 `README.md` 列出全部脚本清单。

> 注：这些脚本依赖教材原始数据，数据已尽量从教材资料压缩包中提取到 `data/` 对应子目录（单文件 >50MB 的未自动提取，请按脚本内路径自行补充）。

---

## 构建工具（`tools/`）

| 脚本 | 用途 |
|------|------|
| `nb_to_py.py` | 单个 `.ipynb` → `.py` 转换器（保留注释） |
| `convert_all_notebooks.py` | 批量从教材 zip 提取 notebook 并转换 + 抽取数据 |
| `download_wheels.py` | 离线环境通过 curl 下载依赖 wheel（pip 被墙时的替代方案） |
| `xlsx_inspect.py` | 标准库解析 xlsx，定位多表 sheet 的表头行 |

---

## 运行环境

- Python 3.11+（建议 3.13）
- 依赖见 `requirements.txt`：
  pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn, mlxtend, openpyxl

```bash
pip install -r requirements.txt
```

---

## 许可证

[MIT](LICENSE) — 教材版权归原作者/出版方所有，本仓库仅作学习与技术复现用途。
