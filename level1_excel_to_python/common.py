#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共工具：定位数据目录、Excel 日期序列号(1899-12-30 基准)转 datetime、
以及统一保存图片。所有一级教材迁移脚本都依赖本模块。
"""
import os
import matplotlib

matplotlib.use("Agg")  # 无界面环境下也能画图并存盘
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 让 matplotlib 支持中文（Windows 自带 SimHei；其他系统若无此字体则降级，不影响代码运行）
for _fp in (r"C:/Windows/Fonts/simhei.ttf", r"/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
    if os.path.exists(_fp):
        try:
            font_manager.fontManager.addfont(_fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

# 仓库根目录：本文件位于 <repo>/level1_excel_to_python/common.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_LEVEL1 = os.path.join(REPO_ROOT, "data", "level1")

# Excel 日期序列号基准（1900 日期系统，1899-12-30 为第 0 天）
EXCEL_EPOCH = 693594  # days between 0001-01-01 and 1899-12-30 in proleptic gregorian


def excel_serial_to_datetime(series):
    """将 Excel 日期序列号(整数/浮点)转换为 pandas datetime。
    若列已经被 openpyxl/pandas 自动解析成 datetime，则原样返回。"""
    import pandas as pd
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")


def save_fig(fig, name, out_dir):
    """保存 matplotlib 图像到 out_dir，返回完整路径。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
