"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 第二章某银行贷款产品精准营销模型代码
# ============================================================

# ============================================================
# 数据说明
# ============================================================

# ============================================================
# 1. 账户表（Accounts）
# ============================================================

# ============================================================
# 2. 客户息表（Clients）
# ============================================================

# ============================================================
# 3.权限分配表（Disp）
# ============================================================

# ============================================================
# 4.支付订单表（Orders）
# ============================================================

# ============================================================
# 5.交易表（Trans）
# ============================================================

# ============================================================
# 6.贷款表（Loans）
# ============================================================

# ============================================================
# 7.信用卡（Cards）
# ============================================================

# ============================================================
# 8.人口地区统计表（District）
# ============================================================

# ============================================================
# 提取被解释变量
# ============================================================

# ============================================================
# 提取静态特征和时点特征
# ============================================================

# ============================================================
# 提取时期特征
# ============================================================

# ============================================================
# 提取预测用数据宽表
# ============================================================

# ============================================================
# 定量客户画像与数据清洗
# ============================================================

# --- code cell ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

# --- code cell ---
snd = pd.read_csv("tran0917.csv")

# --- code cell ---
snd.describe().T

# ============================================================
# 由于有缺失值的这些变量可以根据业务判断，该客户没有发生这个业务，因此可以用“0”代替
# ============================================================

# --- code cell ---
snd['avg_balance']= snd['avg_balance'].fillna(0)
snd['std_balance']= snd['std_balance'].fillna(0)
snd['cv_balance']= snd['cv_balance'].fillna(0)
snd['sum_expenditure']= snd['sum_expenditure'].fillna(0)
snd['sum_income']= snd['sum_income'].fillna(0)
snd['in_out_ratio']= snd['in_out_ratio'].fillna(0)

# --- code cell ---
sns.boxplot(x='y',y='cv_balance',data=snd)

# ============================================================
# 盖帽法是最常用且最方便的异常值清洗方法，即设定一个上下限，小于下限和大于上限的数值，均用下限和上限数值替代
# ============================================================

# --- code cell ---
#定义盖帽法异常值清洗函数
def blk(floor, root): # 'blk' will return a function
    def f(x):       
        if x < floor:
            x = floor
        elif x > root:
            x = root
        return x
    return f

# 使用该变量的1和99百分位数作为清洗数据的上下限
q1 = snd['std_balance'].quantile(0.01) # 计算百分位数
q99 = snd['std_balance'].quantile(0.99)
blk_tot = blk(floor=q1, root=q99) # 'blk_tot' is a function
snd['std_balance']= snd['std_balance'].map(blk_tot)

# ============================================================
# 建立逻辑回归模型
# ============================================================

# --- code cell ---
train = snd.sample(frac=0.7, random_state=1234).copy()
test = snd[~ snd.index.isin(train.index)].copy()
print(' 训练集样本量: %i \n 测试集样本量: %i' %(len(train), len(test)))

# --- code cell ---
import statsmodels.formula.api as smf
lg = smf.logit('y ~ avg_balance+cv_balance+sum_expenditure+C(card)', data=train).fit()
lg.summary()

# ============================================================
# 评估模型
# ============================================================

# --- code cell ---
#1.生成预测值
train['proba'] = lg.predict(train)
test['proba'] = lg.predict(test)

#2.计算绘制ROC曲线需要的指标
import sklearn.metrics as metrics
fpr_test, tpr_test, th_test = metrics.roc_curve(test.y, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.y, train.proba)

#3.绘制ROC曲线
plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()

# --- code cell ---
#4.打印测试数据集的ROC曲线下面积
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

# ============================================================
# 模型运用的准备工作
# ============================================================

# --- code cell ---
pred = pd.read_csv("pred0917.csv")
#模型中参与建模的变量进行填补缺失
pred['avg_balance']= pred['avg_balance'].fillna(0)
pred['cv_balance']= pred['cv_balance'].fillna(0)
pred['sum_expenditure']= pred['sum_expenditure'].fillna(0)

#对cv_balance进行异常值处理
q1 = pred['cv_balance'].quantile(0.01) # 计算百分位数
q99 = pred['cv_balance'].quantile(0.99)
blk_tot = blk(floor=q1, root=q99) # 'blk_tot' is a function
pred['cv_balance']= pred['cv_balance'].map(blk_tot)

#对sum_expenditure进行异常值处理
q1 = snd['sum_expenditure'].quantile(0.01) # 计算百分位数
q99 = snd['sum_expenditure'].quantile(0.99)
blk_tot = blk(floor=q1, root=q99) # 'blk_tot' is a function
snd['sum_expenditure']= snd['sum_expenditure'].map(blk_tot)

# --- code cell ---

