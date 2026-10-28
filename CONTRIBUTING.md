# 贡献指南（CONTRIBUTING）

感谢关注本仓库！这里收录的是 CDA《商业数据分析》教材实验的 Python 化成果。

## 运行环境
- Python 3.11+
- 安装依赖：`pip install -r requirements.txt`
- 建议在隔离环境（venv / conda）中安装，避免污染全局

## 目录约定
- `level1_excel_to_python/`：一级教材 Excel / Power BI 实验的 **Python 迁移**（手写重写，要求可独立运行并出图）
- `level2_quant_strategy_python/` ~ `level4_finance_cases_python/`：二~四级教材 Jupyter notebook 经 `tools/nb_to_py.py` 转换得到的 `.py`
- `data/`：实验数据集（超大文件已配置 Git LFS）
- `tools/`：构建与转换脚本（notebook→py、离线 wheel 下载等）

## 提交规范
- 一级迁移脚本务必保持「可独立运行 + 出图」，请在本地用 `python xxx.py` 验证通过后再提交
- notebook 转换脚本**不要手工改动业务逻辑**，如需重新生成，统一通过 `tools/convert_all_notebooks.py`
- 新增第三方库依赖时，同步更新 `requirements.txt` 并锁定版本

## Issue / PR
- 提交 Bug 请附：复现步骤、关键报错、对应教材章节
- 提交 PR 请说明：对应级别 / 章节、改动点、验证方式
