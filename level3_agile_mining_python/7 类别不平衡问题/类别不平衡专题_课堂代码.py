"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
# pip install imbalanced-learn

# --- code cell ---
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd

# ============================================================
# # 生成不平衡数据集
# ============================================================

# --- code cell ---
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
class_1 = 1000       #类别1有1000个样本
class_2 = 100        #类别2只有100个
centers = [[0,0], [2.0, 2.0]]         #设定两个类别的中心
clusters_std = [1.5, 1]             #设定两个类别的标准差，通常来说，样本量比较大的类别会更加松散
X, y = make_blobs(n_samples= [class_1, class_2],
                  centers= centers,
                  cluster_std= clusters_std,
                  random_state= 42, shuffle= False)
# 画图看数据集，其中红色点是少数类，紫色点是多数类
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = "rainbow",s = 10)
# 查看不平衡的样本数量
pd.DataFrame(y).value_counts()

# ============================================================
# # Over-sampling 过采样
# 最简单的过采样方法是对少数类别样本做简单复制，例如sklearn库中的calss_weight参数采用的就是此方法。  
# 进阶的过采样方法为插值法（interpolation），主要有**SMOTE**和**ADASYN**两种方法。需要注意的是，插值法可能会在多数类样本中间插值出一个少数类样本导致类别重叠。  
# ## SMOTE
# 全称Synthetic Minority Oversampling Technique.  
# 算法如下：  
# 1. 计算采样比例N，（默认N= 多数样本量/少数样本量）
# 2. 对每一个少数类样本$x_i$，从其k近邻个少数类样本中有放回地随机抽样N-1个，记为$x_{ik}$。  
# 3. 分别与原样本$x_i$按照如下的公式生成新的插值样本  
# $$x_{new}=x_i+rand(0,1)*(x_i-x_{ik})$$  
#
# 重复进行，直到满足停止条件。  
# 参考文献：Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16:321–357, 2002.  
# ============================================================

# --- code cell ---
# SMOTE过采样
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = 42, sampling_strategy= 0.5)
X_sm, Y_sm = smote.fit_resample(X, y)

# 查看采样后的比例
pd.DataFrame(Y_sm).value_counts()

# 画图看过采样后的数据集
plt.scatter(X_sm[:, 0], X_sm[:, 1], c = Y_sm, cmap = "rainbow",s = 10)

# ============================================================
# # Under-sampling欠采样
# ## Controlled under-sampling
# 常用的是Random under-sampling，即随机删除多数类样本。这是唯一一种可以精确设置欠采样样本数量的常用欠采样算法，因此冠以Controlled名号。
# ============================================================

# --- code cell ---
# Random欠采样
from imblearn.under_sampling import RandomUnderSampler

rus= RandomUnderSampler(sampling_strategy=0.3, random_state= 42)
X_rus, Y_rus = rus.fit_resample(X, y)

# 查看采样后的比例
pd.DataFrame(Y_rus).value_counts()

# 画图看过采样后的数据集
plt.scatter(X_rus[:, 0], X_rus[:, 1], c = Y_rus, cmap = "rainbow",s = 10)

# ============================================================
# ## Cleaning under-sampling
# Cleaning类的欠采样算法无法精确控制欠采样样本的数量
# ### TomekLinks
# 当多数类的某样本y和少数类的某样本x，与另外任意样本z，之间的距离d()存在以下关系
# $$d(x,y)<d(x,z)\qquad and\qquad d(x,y)<d(y,z)$$
# 则称x与y之间存在Tomek Link，此时算法可删除多数类样本y以实现欠采样。  
# 可以发现，TomekLinks算法不适合对有清晰边界的两类样本做欠采样。  
# 参考文献：Ivan Tomek. Two modifications of cnn. IEEE Trans. Systems, Man and Cybernetics, 6:769–772, 1976.
# ============================================================

# --- code cell ---
# TomekLinks欠采样
from imblearn.under_sampling import TomekLinks

tl= TomekLinks()
X_tl, Y_tl= tl.fit_resample(X, y)

# 查看采样后的比例
pd.DataFrame(Y_tl).value_counts()

# 画图看过采样后的数据集
plt.scatter(X_tl[:, 0], X_tl[:, 1], c = Y_tl, cmap = "rainbow",s = 10)

# ============================================================
# ### Edited nearest neighbours
# 算法思路是应用最近邻算法来编辑(edit)数据集, 找出那些与邻居不太友好的样本然后移除。  
# 具体来说，对每一个多数类样本$x_i$，若其k近邻的全部/大多数样本是多数类，则该多数类样本$x_i$会被保留，否则会被移除。  
# 参考文献：Dennis L Wilson. Asymptotic properties of nearest neighbor rules using edited data. IEEE Transactions on Systems, Man, and Cybernetics, pages 408–421, 1972.
# ============================================================

# --- code cell ---
# ENN欠采样
from imblearn.under_sampling import EditedNearestNeighbours

enn= EditedNearestNeighbours(n_neighbors= 20)  # 改变n_neighbors=3 参数以观察欠采样结果
X_enn, Y_enn= enn.fit_resample(X, y)

# 查看采样后的比例
pd.DataFrame(Y_enn).value_counts()

# 画图看过采样后的数据集
plt.scatter(X_enn[:, 0], X_enn[:, 1], c = Y_enn, cmap = "rainbow",s = 10)

# ============================================================
# # Combination sampling联合采样
# ============================================================

# ============================================================
# 如果多数类样本有离群值，SMOTE等过采样方法容易在多数类样本中间错误地插入少数类插值，导致类别重叠。这些问题插值可以通过联合使用欠采样方法来解决。  
# 这里我们分别将Tomeks link和Edited nearest-neighbours联接在SMOTE之后，来清除过采样生成的问题插值，以得到一个尽量干净的数据空间。
# ============================================================

# --- code cell ---
# SMOTE叠加TomekLink
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state= 42)
X_smote_tomek, Y_smote_tomek= smote_tomek.fit_resample(X, y)

# 查看采样后的比例
pd.DataFrame(Y_smote_tomek).value_counts()

# 画图看过采样后的数据集
plt.scatter(X_smote_tomek[:, 0], X_smote_tomek[:, 1], c = Y_smote_tomek, cmap = "rainbow",s = 10)

# --- code cell ---
# SMOTE叠加ENN
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state= 42)
X_smote_enn, Y_smote_enn= smote_enn.fit_resample(X, y)

# 查看采样后的比例
pd.DataFrame(Y_smote_enn).value_counts()

# 画图看过采样后的数据集
plt.scatter(X_smote_enn[:, 0], X_smote_enn[:, 1], c = Y_smote_enn, cmap = "rainbow",s = 10)

# ============================================================
# # 实践问题
# ============================================================

# ============================================================
# ## 不做过采样
# ============================================================

# --- code cell ---
# 读取数据
raw_data= pd.read_csv('train_set.csv')

# --- code cell ---
raw_data.info()

# --- code cell ---
df_features= raw_data.drop(['ID'], axis=1)

# --- code cell ---
df= df_features.dropna().reset_index(drop= True)  # 缺失值
labels= df.pop("y")  # 分割标签
cat_cols= df.select_dtypes(include= ["object"])   # 分类型
num_cols= df.select_dtypes(exclude= ["object"])   # 数值型
# 数据编码
from sklearn.preprocessing import OrdinalEncoder
encoder= OrdinalEncoder()
cat_trans= encoder.fit_transform(cat_cols)
df_cat= pd.DataFrame(cat_trans, columns= cat_cols.columns)
# 数据合并
data= pd.concat([df_cat, num_cols], axis=1)
# 分割测试集
from sklearn.model_selection  import train_test_split
Xtrain, Xtest, Ytrain, Ytest= train_test_split(data, labels, test_size= 0.3, stratify= labels, random_state= 42)

# --- code cell ---
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

train = xgb.DMatrix(data= Xtrain, label= Ytrain)
XGB= xgb.train(params= {}, dtrain= train)

test = xgb.DMatrix(data= Xtest)
print(classification_report(Ytest, XGB.predict(test)>0.5))
print("AUC: ", roc_auc_score(Ytest, XGB.predict(test)))

# ============================================================
# ## 过采样
# ============================================================

# --- code cell ---
# SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = 42)
X_sm, Y_sm = smote.fit_resample(Xtrain, Ytrain)

# 查看采样后的比例
pd.DataFrame(Y_sm).value_counts()

# --- code cell ---
train_sm = xgb.DMatrix(data= X_sm, label= Y_sm)
XGB= xgb.train(params= {}, dtrain= train_sm)

test_sm = xgb.DMatrix(data= Xtest)
print(classification_report(Ytest, XGB.predict(test_sm)>0.5))
print("AUC: ", roc_auc_score(Ytest, XGB.predict(test_sm)))

# --- code cell ---
# 思考：上述过采样得到的数据，是否有问题

# --- code cell ---
# SMOTENC
from imblearn.over_sampling import SMOTENC

smote_nc= SMOTENC(categorical_features= list(cat_cols.columns))
X_smnc, Y_smnc = smote_nc.fit_resample(Xtrain, Ytrain)

# 查看采样后的比例
pd.DataFrame(Y_smnc).value_counts()

# --- code cell ---
train_smnc = xgb.DMatrix(data= X_smnc, label= Y_smnc)
XGB= xgb.train(params= {}, dtrain= train_smnc)

print(classification_report(Ytest, XGB.predict(test_sm)>0.5))
print("AUC: ", roc_auc_score(Ytest, XGB.predict(test_sm)))

# ============================================================
# ## 通过参数解决（XGBoost）
# ============================================================

# ============================================================
# 如果关心预测的整体性能指标 (AUC)，可以使用过采样。例如SMOTE等方法。
#
# XGBoost提供了参数scale_pos_weight= 1。正类样本的损失会乘以该权重，从而加大其在梯度计算中的贡献。更有效地解决类别不平衡问题
# ============================================================

# --- code cell ---
weight= Ytrain.value_counts()[0] / Ytrain.value_counts()[1]

params= {'scale_pos_weight': weight, 'eval_metric': 'auc'}

XGB= xgb.train(params= params, dtrain= train)

print(classification_report(Ytest, XGB.predict(test)>0.5))
print("AUC: ", roc_auc_score(Ytest, XGB.predict(test)))

# ============================================================
# 如果关心预测正确的概率，此时若重新平衡数据集，会改变标签的分布，可能会影响模型投产后的预测效果。在这种情况下，不推荐重新平衡数据集。
#
# XGBoost提供了参数max_delta_step= 0。通过控制控制权重更新步长的范围，来防止类别不平衡导致的极端的梯度更新
# ============================================================

# --- code cell ---
params= {'max_delta_step': 0.6, 'eval_metric': 'auc'}

XGB= xgb.train(params= params, dtrain= train)

print(classification_report(Ytest, XGB.predict(test)>0.5))
print("AUC: ", roc_auc_score(Ytest, XGB.predict(test)))

# --- code cell ---


# ============================================================
# LightGBM虽然也提供了类似参数，但LightGBM更依赖学习率和正则化参数（如lambda_l2）来控制梯度更新的幅度和训练稳定性，而不直接使用类似于 max_delta_step 的参数。
# ============================================================

# --- code cell ---

