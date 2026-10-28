"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 逻辑回归建模案例
# ============================================================

# ============================================================
# |字段名|中文含义|
# |:--:|:--:|
# |id|id|
# |Acc|是否开卡(1=已开通)|
# |avg_exp|月均信用卡支出（元）|
# |avg_exp_ln|月均信用卡支出的自然对数|
# |gender|性别(男=1)|
# |Age|年龄|
# |Income|年收入（万元）|
# |Ownrent|是否自有住房（有=1；无=0)|
# |Selfempl|是否自谋职业(1=yes, 0=no)|
# |dist_home_val|所住小区房屋均价(万元)|
# |dist_avg_income|当地人均收入|
# |high_avg|高出当地平均收入|
# |edu_class|教育等级：小学及以下开通=0，中学=1，本科=2，研究生=3|
# ============================================================

# --- code cell ---
# %matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

pd.set_option('display.max_columns', 8)

# --- code cell ---
data = pd.read_csv(r'creditcard_exp.csv', skipinitialspace=True)
data.head()

# --- code cell ---
oos_data = pd.read_csv(r'creditcard_exp_out_of_sample.csv', skipinitialspace=True)
oos_data.head()

# --- code cell ---
data = data.drop('avg_exp', axis=1)\
           .drop('avg_exp_ln', axis=1)\
           .drop('age2', axis=1)
oos_data = oos_data.drop('age2', axis=1)
oos_data['edu_class'].replace({0: 1}, inplace=False)

# ============================================================
# 拆分训练集和测试集
# ============================================================

# --- code cell ---
train = data.sample(frac=0.7, random_state=1234).copy()
test = data[~ data.index.isin(train.index)].copy()
print(' 训练集样本量: %i \n 测试集样本量: %i' %(len(train), len(test)))

# ============================================================
# ### 使用statsmodels进行逻辑回归建模
#
# ** 常用的机器学习、统计分析框架均支持逻辑回归，本章后续使用更流行的scikit-learn进行建模和优化*
# ============================================================

# --- code cell ---
lr_model = smf.logit(
    formula='Acc ~ Income ', 
    data=train, 
#     family=sm.families.Binomial(sm.families.links.logit)
).fit()
lr_model.summary()

# --- code cell ---
test['Pred_Acc'] = lr_model.predict(test)
print(test['Pred_Acc'].head())

# --- code cell ---
#  不同阈值下的模型评估

for i in np.arange(0.1, 1, 0.2):
    prediction = (test['Pred_Acc'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction,test['Acc'],
                                   margins=True)
    precision = confusion_matrix.iloc[0, 0] /confusion_matrix.loc['All', 0]
    recall = confusion_matrix.iloc[0, 0] / confusion_matrix.loc[0, 'All']
    Specificity = confusion_matrix.iloc[1, 1] /confusion_matrix.loc[1,'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('Threshold: %.2f' %(i))
    print('Precision: %.2f, Recall:%.2f ,Specificity:%.2f ,f1_score:%.2f \n' \
          %(precision, recall, Specificity,f1_score))

# ============================================================
# ### 变量筛选方案一：WOE转换 
# - 同时对变量进行WOE编码
# # 如果使用WOE包，需要将pandas降级到1.5版本。因此本方法不建议使用了。后续课程中会使用toad包做WOE转换
#
# *本章后续建模过程采用scikit-learn*
# ============================================================

# ============================================================
# from woe import WoE
# ============================================================

# ============================================================
# **WoE类参数说明**:
# + ```qnt_num:int```,等频分箱个数,默认16
# + ```min_block_size:int```,最小观测数目，默认16
# + ```spec_values:dict```,若为分类自变量，指派替换值
# + ```v_type:str```,自变量类型,分类:‘d’,连续变量:‘c’，默认'c'
# + ```bins:list```,预定义的连续变量分箱区间
# + ```t_type:str```,目标变量类型,二分类:‘b’,连续变量:‘c’，默认'b'
#
# **WoE类重要方法**:
#
# + ```plot```:绘制WOE图
# + ```transform```:转换数据为WOE数据
# + ```fit_transform```:转换数据为WOE数据
# + ```optimize```:连续变量使用最优分箱
#
# **WoE类重要属性**:
# + ```bins```:分箱结果汇总
# + ```iv```:变量的信息价值
# ============================================================

# ============================================================
# y_col = 'Acc'
# x_col_continuous = ['Age', 'Income', 'dist_home_val', 'dist_avg_income', 'high_avg']
# x_col_discrete = ['gender', 'Ownrent', 'Selfempl', 'edu_class']
# x_col_all = x_col_continuous + x_col_discrete
# ============================================================

# ============================================================
# #  对离散变量计算woe和IV
# iv_d = {}
# for x_col in x_col_discrete:
#     _woe = WoE(v_type='d',t_type='b')
#     _woe.fit(train[x_col],train[y_col])
#     #  用woe编码替换原始变量
#     train[x_col] = _woe.transform(train[x_col])['woe']
#     test[x_col] = _woe.transform(test[x_col])['woe']
#     oos_data[x_col] = _woe.transform(oos_data[x_col])['woe']
#     #  保存IV值
#     iv_d[x_col] = _woe.iv
#
# print(pd.Series(iv_d).sort_values(ascending=False))
# ============================================================

# ============================================================
# #  对连续变量进行分箱、woe编码、计算IV
# iv_c = {}
# for x_col in x_col_continuous:
#     _woe = WoE(v_type='c',t_type='b', qnt_num=20)
#     _woe.fit(train[x_col],train[y_col])
#
#     # 使用决策树对连续变量进行分箱的优化
#     _woe = _woe.optimize(max_depth=5)
#
#     #  保存IV值
#     iv_c[x_col] = _woe.iv
# print(pd.Series(iv_d).sort_values(ascending=False))
# print('-' * 30)
# print(pd.Series(iv_c).sort_values(ascending=False))
# ============================================================

# ============================================================
# ### 变量筛选方案二——递归特征消除（向后法）
# - 使用Scikit-learn中的RFE（Recursive Feature Elimination）
# ============================================================

# ============================================================
# #### 先完成数据预处理
# 1. 连续变量：标准化、离散化
# 2. 离散变量：哑变量编码（one_hot编码）、woe编码
# ============================================================

# --- code cell ---
y_col = 'Acc'
x_col_continuous = ['Age', 'Income', 'dist_home_val', 'dist_avg_income', 'high_avg']
x_col_discrete = ['gender', 'Ownrent', 'Selfempl', 'edu_class']
x_col_all = x_col_continuous + x_col_discrete

# --- code cell ---
#  连续变量标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train[x_col_continuous] = scaler.fit(train[x_col_continuous])\
                                .transform(train[x_col_continuous])

#  对测试集连续变量做同样的标准化
test[x_col_continuous]  = scaler.transform(test[x_col_continuous])
oos_data[x_col_continuous] = scaler.transform(oos_data[x_col_continuous])

# --- code cell ---
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train[x_col_all], train[y_col])

# 打印测试集预测结果
print('Prediction of test: ')
print(lr.predict(test[x_col_all]))

# --- code cell ---
# 打印测试集预测概率
print('Probability of test: ')
print(lr.predict_proba(test[x_col_all]))

# ============================================================
# #### 带交叉验证的递归特征消除 （RFECV：Recursive Feature Elimination and Cross-Validated selection）
# ============================================================

# --- code cell ---
from sklearn.feature_selection import RFECV

rfecv = RFECV(lr, cv=3, scoring="accuracy",n_jobs=-1)
rfecv.fit(train[x_col_all], train[y_col])

# RFECV.support_返回保留下来的特征的mask
selected_cols = np.array(x_col_all)[rfecv.support_]

print('Ranking of columns: ', rfecv.ranking_)
print('Selected columns: ', selected_cols)

# ============================================================
# ### 交叉验证选择超参
# ============================================================

# --- code cell ---
from sklearn.linear_model import LogisticRegressionCV

 # Cs为正则化系数的倒数
lrcv = LogisticRegressionCV(Cs=10, cv=4, n_jobs=-1)  
lrcv.fit(train[selected_cols], train[y_col])
print('Regulation strength: ', lrcv.C_)  # 模型交叉验证分数最高的C值

# --- code cell ---
#  预测标签
train_prediction = lrcv.predict(train[selected_cols])
test_prediction = lrcv.predict(test[selected_cols])

#  预测概率分值
train_proba = lrcv.predict_proba(train[selected_cols])[:,1]
test_proba = lrcv.predict_proba(test[selected_cols])[:,1]

# ============================================================
# ### scikit-learn中的模型评估
# ============================================================

# ============================================================
# - 混淆矩阵
# ============================================================

# --- code cell ---
from sklearn.metrics import roc_curve, classification_report, auc

print(classification_report(test['Acc'], test_prediction))  # 计算评估指标

# ============================================================
# - ROC曲线与AUC
# ============================================================

# --- code cell ---
#  生成TPR和FPR
fpr_test, tpr_test, th_test = roc_curve(test['Acc'], test_proba)
fpr_train, tpr_train, th_train = roc_curve(train['Acc'], train_proba)

#  绘制ROC曲线
fig, ax = plt.subplots(figsize=[5, 5])
ax.set_xlabel('FPR',  fontsize=15)
ax.set_ylabel('TPR',  fontsize=15)
ax.plot(fpr_test, tpr_test, 'b-')
ax.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve', fontsize=20)
plt.show()

#  打印AUC
print('AUC on test = %6.4f' % auc(fpr_test, tpr_test))

# ============================================================
# #### 保存预测结果
# ============================================================

# --- code cell ---
oos_data.head()

# --- code cell ---
output_data_dir = './data_processed'
if not os.path.exists(output_data_dir):
    os.mkdir(output_data_dir)
output_data_path = os.path.join(output_data_dir,
                               'creditcard_exp_logistic_regression.csv')

oos_data['Pred_Acc'] = lrcv.predict_proba(oos_data[selected_cols])[:,1]
oos_data.to_csv(output_data_path, index=False)

# ============================================================
# #### 保存和读取模型
# ============================================================

# --- code cell ---
import joblib

output_dir = "./model"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
save_path = os.path.join(output_dir, "creditcard_logisticRegression.model")

# 模型持久化
joblib.dump(lrcv, save_path)
# 读取模型
model_reload = joblib.load(save_path)
model_reload.predict(oos_data[selected_cols])

# --- code cell ---


# --- code cell ---

