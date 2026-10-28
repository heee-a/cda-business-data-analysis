"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# 3.1.3 连续变量的分布与集中趋势
# ============================================================

# --- code cell ---
import pandas as pd
data = pd.read_csv('sndHsPr.csv')

# --- code cell ---
data.price.agg(['mean','median','std'])

# --- code cell ---
data. price.quantile([0.25,0.5,0.75])

# --- code cell ---
data.price.hist(bins=20)

# ============================================================
# 3.1.4 连续变量的离散程度
# ============================================================

# --- code cell ---
data.price.max()-data.price.min()

# --- code cell ---
data.price.var()#求方差

# --- code cell ---
data.price.std()#求标准差

# ============================================================
# 3.1.5 数据分布的对称与高矮
# ============================================================

# --- code cell ---
import numpy as np

# --- code cell ---
normal = pd.Series(np.random.randn(1000),name='normal')

# --- code cell ---
normal.skew()

# --- code cell ---
normal.kurtosis()

# --- code cell ---
data[['AREA','price']].corr("pearson")

# --- code cell ---
data[['AREA','price']].corr("spearman")

# --- code cell ---
data[['AREA','price']].corr("kendall")

# --- code cell ---


# --- code cell ---

