"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
# !pip install lightgbm

# --- code cell ---
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection  import train_test_split
import numpy as np
import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")
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
# # Sklearn部分实现LightGBM
# ============================================================

# --- code cell ---
from sklearn.ensemble import HistGradientBoostingClassifier

HGBC= HistGradientBoostingClassifier()
HGBC.fit(Xtrain, Ytrain)

print("AUC: ", roc_auc_score(Ytest, HGBC.predict_proba(Xtest)[:, 1]))

# ============================================================
# # LightGBM库
# ============================================================

# ============================================================
# ## 基础使用
# ============================================================

# --- code cell ---
# sklearn接口（不推荐）
from lightgbm import LGBMClassifier
lgb= LGBMClassifier()
lgb.fit(Xtrain, Ytrain)

print("AUC: ", roc_auc_score(Ytest, lgb.predict_proba(Xtest)[:, 1]))

# --- code cell ---
# lgb接口
import lightgbm

train= lightgbm.Dataset(Xtrain, label= Ytrain)

lgb = lightgbm.train(params={}, train_set= train)

print("AUC: ", roc_auc_score(Ytest, lgb.predict(Xtest)))

# --- code cell ---
# 数据集转换为二进制保存，加快加载速度
train.save_binary('train.bin')

# --- code cell ---
train1= lightgbm.Dataset('train.bin')

lgb1 = lightgbm.train(params={}, train_set= train1)

# --- code cell ---
# # 附加参数
# metric= ""   评估指标{rmse, quantile, poisson, auc, binary_loglogg, multi_logloss}
# device_type= 'cpu'   {'cpu', 'gpu', 'cuda'}
# max_cat_to_onehot= 4, 自动哑变量转换的类别数
# use_quantized_grad= False,  训练时是否使用梯度量化, 只支持cpu,cuda

# ============================================================
# ## 进阶使用
# ============================================================

# ============================================================
# ### 数据集
# ============================================================

# --- code cell ---
# 数据集
Xtrain1, Xtest1, Ytrain1, Ytest1= train_test_split(Xtrain, Ytrain, test_size= 0.3, stratify= Ytrain, random_state= 42)

train1= lightgbm.Dataset(data= Xtrain1, label= Ytrain1,
                         categorical_feature= ['GEND', 'AART'],
                         params= {'max_bin': 255},   # 附加参数
                         free_raw_data=False)
test1= lightgbm.Dataset(data= Xtest1, label= Ytest1, reference=train1)  # reference参数只能传入训练集

# --- code cell ---
# # 附加参数（数据集）
# max_bin= 255   最大分箱数
# max_bin_by_feature= None   每个特征的最大箱数
# min_data_in_bin= 3   每个箱子内的最小样本数
# bin_construct_sample_cnt= 200000   构造箱子的最大样本数
# two_round= False   数据集两轮加载，只支持cpu,cuda

# ============================================================
# ### 训练
# ============================================================

# --- code cell ---
# 训练
params= {'num_leaves': 31, 'objective': 'binary'}
params['metric'] = 'auc'
params['verbosity'] = 3

lgb = lightgbm.train(params=params, train_set= train1, valid_sets= test1,
                     num_boost_round= 200,
                     callbacks=[lightgbm.early_stopping(stopping_rounds=5, min_delta=0)],   # min_delta至少提升多少数值
                     )

print("AUC: ", roc_auc_score(Ytest, lgb.predict(Xtest)))

# --- code cell ---
# 常用接口
lgb.best_iteration
lgb.feature_importance()

# --- code cell ---
# 模型保存
lgb.save_model('model.txt')

# --- code cell ---
# 单调约束
my_list= [1, 0, -1] + [0] * 46

params= {'metric': 'auc', 'objective': 'binary'}
params['monotone_constraints']= my_list   # 必须传入所有特征的约束标识
params['monotone_constraints_method']= 'advanced'   # {basic, intermediate, advanced}


lgb = lightgbm.train(params=params, train_set= train1, valid_sets= test1)

print("AUC: ", roc_auc_score(Ytest, lgb.predict(Xtest)))

# --- code cell ---
# 交互约束
params= {'metric': 'auc', 'objective': 'binary'}
params['interaction_constraints']= [[1, 3, 4], [5, 6]]

lgb = lightgbm.train(params=params, train_set= train1, valid_sets= test1)

print("AUC: ", roc_auc_score(Ytest, lgb.predict(Xtest)))
