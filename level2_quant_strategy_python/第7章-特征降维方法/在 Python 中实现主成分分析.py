"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 在 Python 中实现主成分分析. 
# 一家电信运营商希望根据客户在几个主要业务上的使用情况，对客户进行分群
# 分析。
# ============================================================

# ============================================================
# ![image.png](attachment:image.png)
# ============================================================

# --- code cell ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- code cell ---
profile = pd.read_csv("profile_telecom.csv")
profile.head()

# --- code cell ---
data = profile.loc[:,"cnt_call":]
data.corr(method = "pearson")

# --- code cell ---
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
data_scaled = scale(data)
pca = PCA(n_components = 2, whiten = True).fit(data_scaled)
pca.explained_variance_ratio_

# --- code cell ---
pca.components_

# --- code cell ---
pca.transform(data_scaled)

# --- code cell ---


# --- code cell ---

