"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# ## 数据读取与处理
# ============================================================

# --- code cell ---
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection  import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# 读入数据
raw_data= pd.read_csv('保险数据_第一期.csv')
# 缺失值(缺失值占比很少，直接删除)
df= raw_data.dropna().reset_index(drop=True)
# 分割标签
labels= df.pop("resp_flag")
# 数据类型拆分
cat_cols= df.select_dtypes(include= ["object"])   # 分类型
num_cols= df.select_dtypes(exclude= ["object"])   # 数值型
# 数据编码
encoder= OrdinalEncoder()
cat_trans= encoder.fit_transform(cat_cols)
df_cat= pd.DataFrame(cat_trans, columns= cat_cols.columns)
# 数据合并
data= pd.concat([df_cat, num_cols], axis=1)
# 分割测试集
Xtrain, Xtest, Ytrain, Ytest= train_test_split(data, labels, test_size= 0.3, stratify= labels, random_state= 42)

# ============================================================
# ## 基础用法
# ============================================================

# --- code cell ---
# sklearn接口
from xgboost.sklearn import XGBClassifier
XGB= XGBClassifier()
XGB.fit(Xtrain, Ytrain)

print("AUC: ", roc_auc_score(Ytest, XGB.predict_proba(Xtest)[:, 1]))

# --- code cell ---
# xgb接口，必须传入专属格式的数据
import xgboost as xgb

train= xgb.DMatrix(data= Xtrain, label= Ytrain)
XGB= xgb.train(params= {}, dtrain= train)

test= xgb.DMatrix(data= Xtest)
print(classification_report(Ytest, XGB.predict(test)>0.5))
print("AUC: ", roc_auc_score(Ytest, XGB.predict(test)))

# --- code cell ---
# 参数设置
param= {'max_depth': 5, 'eta': 0.4}
param['eval_metric'] = 'auc'

XGB= xgb.train(params= param, dtrain= train, evals= [(train, 'train')], num_boost_round= 15)
print("AUC: ", roc_auc_score(Ytest, XGB.predict(test)))

# --- code cell ---
# evals=None   计算评估指标的数据集
# early_stopping_rounds=None   提前停止。int。连续多少次指标没有改进，则返回上次迭代的模型

# --- code cell ---
# 提前停止
Xtrain1, Xtest1, Ytrain1, Ytest1= train_test_split(Xtrain, Ytrain, test_size= 0.3, stratify= Ytrain, random_state= 42)

train_e= xgb.DMatrix(data= Xtrain1, label= Ytrain1)
test_e= xgb.DMatrix(data= Xtest1, label= Ytest1)

# 参数设置
param= {'max_depth': 5, 'eta': 0.4}
param['eval_metric'] = 'auc'

eval_list= [(train_e, 'train_e'), (test_e, 'test_e')]   # 只有最后一个数据集用来计算early stop

XGB= xgb.train(params= param, dtrain= train_e, evals= eval_list, 
               num_boost_round= 15, early_stopping_rounds= 3)

# ============================================================
# ## 进阶用法
# ============================================================

# ============================================================
# ### 单调约束
# ============================================================

# ============================================================
# ![image.png](attachment:image.png)
# ============================================================

# ============================================================
# ![image.png](attachment:image.png)
# ============================================================

# ============================================================
# 如果tree_method设置为hist或approx，启用单调约束可能会产生不必要的浅树。这是因为该hist方法减少了每次分割时要考虑的候选分割数量。单调约束可能会消除所有可用的分割候选，在这种情况下不会进行分割。为了减少这种影响，您可能需要增加max_bin参数或使用exact方法。
# ============================================================

# --- code cell ---
param

# --- code cell ---
# 单调约束
params_monotonic= param.copy()
params_monotonic['tree_method']= 'exact'
params_monotonic['monotone_constraints']= (0, 1, -1)   # 前三个特征分别：不设约束，单调增，单调减

XGB= xgb.train(params= params_monotonic, dtrain= train, evals= [(train, 'train')])
print("AUC: ", roc_auc_score(Ytest, XGB.predict(test)))

# ============================================================
# ### 交互约束
# ============================================================

# ============================================================
# ![image.png](attachment:image.png)
# ============================================================

# ============================================================
# 特征交互约束以允许交互的特征list来表示。例如，约束[0, 1]表示特征0和1允许相互作用，但不允许其他特征相互作用。同理，[2, 3, 4]表示特征2,3,4允许彼此交互，但不能与其他特征交互。
# ============================================================

# --- code cell ---
# 交互约束
params_interaction= param.copy()
params_interaction['interaction_constraints']= '[[0, 2], [1, 3, 4], [5, 6]]'

XGB= xgb.train(params= params_interaction, dtrain= train, evals= [(train, 'train')])
print("AUC: ", roc_auc_score(Ytest, XGB.predict(test)))

# ============================================================
# ### 基于XGBoost的随机森林（类似）
# ============================================================

# --- code cell ---
# 随机森林
params_rf= {
          "colsample_bynode": 0.8,   # colsample类型的参数设置任何一个即可，必须小于1
          "learning_rate": 1,   # 必须设为1
          "num_parallel_tree": 100,   # 森林中树的数量
          "subsample": 0.8,   # 必须小于1
            }

XGB= xgb.train(params= params_rf, dtrain= train, evals= [(train, 'train')], num_boost_round= 1)   # 迭代次数必须设为1，否则会生成多个森林
print("AUC: ", roc_auc_score(Ytest, XGB.predict(test)))
