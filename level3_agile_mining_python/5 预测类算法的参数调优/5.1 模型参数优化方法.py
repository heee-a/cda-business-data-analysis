"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 数据处理
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
x_train, x_test, y_train, y_test= train_test_split(data, labels, test_size= 0.3, stratify= labels, random_state= 42)

# ============================================================
# # Tuning the hyper-parameters（调参）
# ============================================================

# ============================================================
# ## Grid Search（网格搜索）
# ============================================================

# --- code cell ---
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

param = {"n_estimators": np.arange(100,150,20),    # 启，止，步长
         "max_depth": np.arange(3,6,2),
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9]}

xgb = XGBClassifier()
grid_search_xgb = GridSearchCV(xgb, param, cv=3, n_jobs=-1)  
grid_search_xgb.fit(x_train, y_train)

print("AUC: ", roc_auc_score(y_test, grid_search_xgb.predict_proba(x_test)[:, 1]))
grid_search_xgb.best_params_ #最优参数

# ============================================================
# ## Halving Search
# ============================================================

# --- code cell ---
# Halving 搜索
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

param = {"n_estimators": np.arange(100,150,1),    # 启，止，步长
         "max_depth": np.arange(3,6,1),
        "subsample": np.arange(0.5,1,0.1),
        "colsample_bytree": np.arange(0.5,1,0.1)}

xgb = XGBClassifier()
halving_search_xgb = HalvingGridSearchCV(xgb, param, factor=3, cv=3, verbose=1, n_jobs=-1)  
halving_search_xgb.fit(x_train, y_train)

print("AUC: ", roc_auc_score(y_test, halving_search_xgb.predict_proba(x_test)[:, 1]))
halving_search_xgb.best_params_ #最优参数

# ============================================================
# ## Randomized Search（随机搜索）
# ============================================================

# --- code cell ---
# 随机搜索
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

param = dict(n_estimators= np.arange(100,150,1),    # 启，止，步长
         max_depth= np.arange(3,6,1),
         subsample= uniform(loc=0.5, scale=0.5),   # 启，增加量
         colsample_bytree= uniform(loc=0.5, scale=0.5))

xgb = XGBClassifier()
random_search_xgb = RandomizedSearchCV(xgb, param, cv=3, n_iter=50, verbose=1, n_jobs=-1)  
random_search_xgb.fit(x_train, y_train)

print("AUC: ", roc_auc_score(y_test, random_search_xgb.predict_proba(x_test)[:, 1]))
random_search_xgb.best_params_ #最优参数

# --- code cell ---
# 查看最优模型参数
random_search_xgb.best_params_

# ============================================================
# ## Bayesian Optimization（贝叶斯搜索）
# ============================================================

# --- code cell ---
# 贝叶斯搜索
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb

def objective(trial):  # 定义优化函数
    param = {
        'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        "verbosity" : -1,   # 静默输出，训练过程不输出冗余信息
    }
    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_test, label=y_test)
    
    model = lgb.train(params= param, train_set=d_train, valid_sets=d_valid) # 建模
    AUC= roc_auc_score(y_test, model.predict(x_test))   # metric
    return AUC  # AUC作为目标值
sampler = TPESampler()  # 使用基于树结构的贝叶斯优化算法-TPE
lgb_opt = optuna.create_study(sampler=sampler, direction='maximize')  # 最大化目标值
lgb_opt.optimize(
    func=objective,   # 要优化的函数
    n_trials=100,     # 100次迭代
    show_progress_bar=True,   # 显示进度条
    n_jobs=-1         # 使用所有CPU核心并行化
)  

# 打印结果
print("Best_params: ", lgb_opt.best_params)
print("Best score: ", lgb_opt.best_value)

# --- code cell ---

