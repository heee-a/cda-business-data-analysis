"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # Pipeline介绍  
# 一、  
# Pipelines ,直译是“管道“， 类似于流水线的意思，可以将数据预处理和建模流程封装起来。
# 在数据处理过程中，很多步骤都是重复或者类似的，比如数据处理、特征选择、标准化、分类等等，pipeline就可以实现以下几点好处：
# 1. 简化代码：直接将步骤封装成完整的工作流，避免了代码重复。
# 2. 更少出Bug：流程规范化，避免在建模和部署过程中漏掉某个步骤。
# 3. 更易于生产/复制：直接调用fit和predict来对管道中所有的算法模型一次性进行训练和预测。
# 4. 简化模型验证过程：网格搜索（Grid Search）可以遍历管道中所有评估器的参数。
#
# Pipeline是使用 （key，value） 对的列表构建的，其中key是步骤名称的字符串，而value是一个估计器对象。
#
# 二、  
# Sklearn中有两个pipeline类型的模块，分别是：  
# 1. ColumnTransformer，主要用于数据与特征处理的工作流，可并行
# 2. pipeline，可用于任何需求的工作流，只能串行
#
# **重要：管道中的所有评估器，除了最后一个评估器，管道的所有评估器必须是转换器。 (例如，必须有 transform 方法). 最后一个评估器的类型不限（转换器、分类器等等）**
# ============================================================

# --- code cell ---
# 导入所有的库
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# # 数据导入与处理（ColumnTransformer）
# ============================================================

# --- code cell ---
# 导入数据，拆分标签
df= pd.read_csv('保险数据_第一期.csv')

labels= df.pop('resp_flag')   # 取出y

# --- code cell ---
# 数据类型区分
cat_cols= df.select_dtypes(include=["object"])   # 分类型变量
num_cols= df.select_dtypes(exclude=["object"])   # 数值型变量

# --- code cell ---
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- code cell ---
# 分类型数据
cat_imp= SimpleImputer(strategy='most_frequent')   # 缺失值
cat_encode= OrdinalEncoder()   # 数据编码
cat_pipe= Pipeline(steps= [('cat_imp', cat_imp), ('cat_encode', cat_encode)])

# 数值型数据
num_imp= SimpleImputer(strategy='mean')   # 缺失值

# ColumnTransformer，List of (name, transformer, columns) tuples
col_trans= ColumnTransformer(transformers= [('cat_pipe', cat_pipe, cat_cols.columns),
                                            ('num_imp', num_imp, num_cols.columns)])

# --- code cell ---
# 测试
col_trans.fit_transform(df)

# ============================================================
# # 实例化模型，建立Pipeline
# ============================================================

# --- code cell ---
# pipeline，List of (name, transform) tuples
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr_pipe= Pipeline(steps= [('col_trans', col_trans), ('lr', lr)])

# ============================================================
# # 训练模型，调参
# ============================================================

# --- code cell ---
# 分割测试集
from sklearn.model_selection  import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(df, labels, test_size=0.3, stratify=labels, random_state= 42)

# --- code cell ---
# 测试
lr_pipe.fit(Xtrain, Ytrain)
lr_pipe.score(Xtest, Ytest)

# --- code cell ---
# 网格搜索，使用<estimator>__<parameter>访问每个transform内部的参数
from sklearn.model_selection import GridSearchCV

# 参数字典
param= dict(lr__C= np.arange(1, 2, 0.1),
           lr__class_weight= [None, "balanced"],
           col_trans__num_imp__strategy= ['mean', 'median'])

grid_lr= GridSearchCV(lr_pipe, param, cv= 3, n_jobs= -1)
grid_lr.fit(Xtrain, Ytrain)

# --- code cell ---
grid_lr.best_params_

# --- code cell ---
# 模型评估
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

print(classification_report(Ytest, grid_lr.predict(Xtest)))
print("AUC: ", roc_auc_score(Ytest, grid_lr.predict_proba(Xtest)[:, 1]))

# --- code cell ---
print(classification_report(Ytrain, grid_lr.predict(Xtrain)))
print("AUC: ", roc_auc_score(Ytrain, grid_lr.predict_proba(Xtrain)[:, 1]))

# ============================================================
# # 模型的保存
# ============================================================

# --- code cell ---
# 保存模型
import joblib
joblib.dump(grid_lr, "model_20241222.joblib")

# ============================================================
# # 预测
# ============================================================

# --- code cell ---
# 预测新进用户
df_new= pd.read_csv("保险案例_新进用户.csv")
df_new

# --- code cell ---
grid_lr.predict(df_new)

# --- code cell ---
# 作业：继续做特征工程，包括加入数据标准化，模型换成决策树并调参，以实现更好的模型结果

# --- code cell ---

