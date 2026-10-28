"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# |变量名|变量含义|
# |:--:|:--:|
# |X1|品格：指客户的名誉|
# |X2|能力：指客户的偿还能力|
# |X3|资本：指客户的财务实力和财务状况|
# |X4|担保：指对申请贷款项担保的覆盖程度|
# |X5|环境：指外部经济、政策环境对客户的影响|
# ============================================================

# --- code cell ---
import pandas as pd
import numpy as np

raw_data = pd.read_csv("Loan_aply.csv",encoding='gbk')
raw_data.head()

# --- code cell ---
data = raw_data.loc[ :,'X1':]
data.corr(method='pearson')

# --- code cell ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
pca = PCA(n_components=5)
pca.fit(scaled_data)

# 输出主成分可解释变异
print('Explained variance: \n', pca.explained_variance_)  
# 输出可解释变异的比例
print('Explained variance ratio:\n', pca.explained_variance_ratio_) 

# --- code cell ---
# 打印主成分向量
print('Principal Components: \n', pca.components_)

# ============================================================
# - 说明：1、第一次的n_components参数应该设的大一点
# - 说明：2、观察explained_variance_ratio_和explained_variance_的取值变化，建议explained_variance_ratio_累积大于0.85，explained_variance_需要保留的最后一个主成分大于0.8
# ============================================================

# --- code cell ---
scaled_data[0]

# --- code cell ---
raw_data[['ID']].join(pd.DataFrame(pca.transform(scaled_data)))

# --- code cell ---
print('PCA_score of "ID=1":  %.4f'
      %np.dot(scaled_data[0], pca.components_[0]))

# --- code cell ---
pca.set_params(n_components=1)
pca.fit(scaled_data)
data_pca = raw_data.join(
    pd.DataFrame(pca.transform(scaled_data), columns=['PCAScore']))

data_pca.head()

# --- code cell ---


# --- code cell ---


# --- code cell ---

