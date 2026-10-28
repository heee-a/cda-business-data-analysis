"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 线性回归模型与诊断
# 数据说明：本数据是一份汽车贷款数据
#
# 假定：
#   - "creditcard_exp.csv"保存了上一期的数据，Acc、avg_exp和avg_exp为观察期的因变量取值，剩余字段为分析窗口（观察期之前）的自变量取值
#   - "creditcard_exp_out_of_sample.csv"保存了本期分析窗口内的自变量，其因变量需要通过模型输出
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
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

pd.set_option('display.max_columns', 8)

# ============================================================
# #### 导入数据和数据清洗
# ============================================================

# --- code cell ---
raw_data = pd.read_csv('creditcard_exp.csv', skipinitialspace=True)
out_of_sample_data = pd.read_csv('creditcard_exp_out_of_sample.csv', skipinitialspace=True)
raw_data.head()

# --- code cell ---
train = raw_data[raw_data['avg_exp'].notnull()].copy().iloc[:, 2:]\
    .drop('age2',axis=1)

oos_data = out_of_sample_data.copy()\
    .drop('age2',axis=1)

train.describe(include='all')

# ============================================================
# #### 相关性分析
# 散点图
# ============================================================

# --- code cell ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylabel(..., fontsize=15)
ax.set_xlabel(..., fontsize=15)
train.plot('Income', 'avg_exp', kind='scatter', ax=ax)
plt.show()

# --- code cell ---
train[['Income', 'avg_exp', 'Age', 'dist_home_val']].corr(method='pearson')

# ============================================================
# ## 1、线性回归算法
# ### 简单线性回归
# ============================================================

# --- code cell ---
simple_linear_model = ols(formula='avg_exp ~ Income', data=train).fit()
print(simple_linear_model.params)

# --- code cell ---
summary = simple_linear_model.summary()
for table in summary.tables:
    print(table)

# --- code cell ---
simple_linear_model.summary()

# ============================================================
# Predict-在原始数据集上得到预测值和残差
# ============================================================

# --- code cell ---
train_prediction = simple_linear_model.predict(train)
train_resid = simple_linear_model.resid
pd.DataFrame(
    [train_prediction, train_resid], 
    index=['Pred_avg_exp', 'resid']
).T.head()

# ============================================================
# 在待预测数据集上得到预测值
# ============================================================

# --- code cell ---
print(simple_linear_model.predict(oos_data)[:5])

# ============================================================
# ### 多元线性回归
# ============================================================

# --- code cell ---
multiple_linear_model = ols(
    'avg_exp ~Income + dist_home_val + dist_avg_income',
    data=train).fit()
multiple_linear_model.summary()

# --- code cell ---
print(multiple_linear_model.params)

# ============================================================
# # 2、多元线性回归的变量筛选
# ============================================================

# --- code cell ---
'''forward select'''
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            aic = ols(formula=formula, data=data).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:        
            print ('forward selection over!')
            break
            
    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = ols(formula=formula, data=data).fit()
    return(model)

# --- code cell ---
data_for_select = train[['avg_exp', 'Income', 'Age', 'dist_home_val', 
                         'dist_avg_income']]
forward_select_model = forward_select(data=data_for_select, response='avg_exp')
print(forward_select_model.rsquared)

# ============================================================
# # 3、线性回归的诊断
# ### 残差分析
# ============================================================

# --- code cell ---
def plot_resid(resid, x, y_hat, **kw):
    plt.figure(figsize=(8, 8))
    
    # “残差-X”散点图
    ax1 = plt.subplot(211)
    ax1.set_ylabel('resid', fontsize=15)
    ax1.set_xlabel('x', fontsize=15)
    ax1.scatter(x, resid, **kw)
    ax1.set_title('Scatter of resid - x', fontsize=15)
    
    # "残差-Y估计”散点图
    ax2 = plt.subplot(212)
    ax2.set_ylabel('resid', fontsize=15)
    ax2.set_xlabel('y_hat', fontsize=15)
    ax2.scatter(y_hat, resid, **kw)
    ax2.set_title('Scatter of resid - y_hat', fontsize=15)
    
    # 输出图形
    plt.tight_layout() 
    plt.show()

# --- code cell ---
train['Income_ln'] = np.log(train['Income'])

# 样本外数据做同样变换
oos_data['Income_ln'] = np.log(oos_data['Income'])  

# 对x和y各自取对数，比较结果
linear_model1 = ols(formula='avg_exp ~ Income', data=train).fit()
linear_model2 = ols(formula='avg_exp_ln ~ Income', data=train).fit()
linear_model3 = ols(formula='avg_exp_ln ~ Income_ln', data=train).fit()

# --- code cell ---
plot_resid(linear_model1.resid, 
           train['Income'],
           linear_model1.predict(train))

# ============================================================
# 遇到异方差情况,教科书上会介绍使用加权最小二乘法，但是实际上最常用的是对被解释变量取对数
# ============================================================

# --- code cell ---
plot_resid(linear_model2.resid, 
           train['Income'],
           linear_model2.predict(train))

# ============================================================
# 取对数会使模型更有解释意义
# ============================================================

# --- code cell ---
plot_resid(linear_model3.resid, 
           train['Income_ln'],
           linear_model3.predict(train))

# --- code cell ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlabel(..., fontsize=15)
ax.set_ylabel(..., fontsize=15)
sm.qqplot(linear_model3.resid, fit=True, line='45', ax=ax, figure=fig)
plt.show()

# ============================================================
# 寻找最优的模型
# ============================================================

# --- code cell ---
import pprint

r_squares = {'exp ~ Income':  linear_model1.rsquared, 
             'ln(exp) ~ Income': linear_model2.rsquared, 
             'ln(exp) ~ ln(Income)': linear_model3.rsquared}
pprint.pprint(r_squares)

# ============================================================
# ### 强影响点分析
# ============================================================

# --- code cell ---
train['resid'] = linear_model3.resid
student_resid = (train['resid'] - train['resid'].mean()
                   ) / train['resid'].std()

# ============================================================
# Find outlier：
# ============================================================

# --- code cell ---
train[np.abs(student_resid) > 2]

# ============================================================
# statemodels包提供了更多强影响点判断指标
# ============================================================

# --- code cell ---
from statsmodels.stats.outliers_influence import OLSInfluence

OLSInfluence(linear_model3).summary_frame().head()

# ============================================================
# Drop outlier
# ============================================================

# --- code cell ---
train = train[np.abs(student_resid) <= 2].copy()
linear_model4 = ols('avg_exp_ln ~ Income_ln', train).fit()

plot_resid(linear_model4.resid, 
           train['Income_ln'],
           linear_model4.predict(train))

# --- code cell ---
linear_model4.rsquared

# ============================================================
# ### 多重共线性分析
# ============================================================

# --- code cell ---
# Step regression is not always work.

# ============================================================
#
# 经过单变量线性回归的处理，我们基本对模型的性质有了一定的了解，接下来可以放入更多的连续型解释变量。在加入变量之前，要注意变量的函数形式转变。比如当地房屋均价、当地平均收入，其性质和个人收入一样，都需要取对数
# ============================================================

# --- code cell ---
# 对解释变量取对数
train['dist_home_val_ln'] = np.log(train['dist_home_val'])
train['dist_avg_income_ln'] = np.log(train['dist_avg_income'])

# 对本期数据也做同样变换
oos_data['dist_home_val_ln'] = np.log(oos_data['dist_home_val'])
oos_data['dist_avg_income_ln'] = np.log(oos_data['dist_avg_income'])

linear_model5 = ols('''avg_exp_ln ~ Age + Income_ln + 
           dist_home_val_ln + dist_avg_income_ln''', train).fit()
linear_model5.summary()

# ============================================================
# The function "statsmodels.stats.outliers_influence.variance_inflation_factor" uses "OLS" to fit data, and it will generates a wrong rsquared. So define it ourselves!
# ============================================================

# --- code cell ---
def vif(df, y_col):
    x_cols = list(df.columns)
    x_cols.remove(y_col)
    formula = y_col + '~' + '+'.join(x_cols)
    rsquared = ols(formula, df).fit().rsquared
    return 1. / (1. - rsquared)

exog = train[['Age', 'Income_ln', 'dist_home_val_ln',
             'dist_avg_income_ln']]

for y_col in exog.columns:
    print('VIF of "%s" is %.2f' %(y_col, vif(df=exog, y_col=y_col)))

# ============================================================
# Income_ln与dist_avg_income_ln具有共线性，使用“高出平均收入的比率”代替其中一个
# ============================================================

# --- code cell ---
train['high_avg_ratio'] = train['high_avg'] / train['dist_avg_income']
oos_data['high_avg_ratio'] = oos_data['high_avg'] / oos_data['dist_avg_income']

# --- code cell ---
exog = train[['Age', 'high_avg_ratio', 'dist_home_val_ln', 
              'dist_avg_income_ln']]

for y_col in exog.columns:
    print('VIF of "%s" is %.2f' %(y_col, vif(df=exog, y_col=y_col)))

# --- code cell ---
var_select_data = train[['avg_exp_ln', 'Age', 'high_avg_ratio', 
                         'dist_home_val_ln', 'dist_avg_income_ln']]
linear_model6 = forward_select(data=var_select_data, 
                               response='avg_exp_ln')
print('r_squared is ', linear_model6.rsquared)

# ============================================================
# ### 增加离散变量
# ============================================================

# --- code cell ---
formula = '''
avg_exp_ln ~ dist_avg_income_ln + dist_home_val_ln + 
C(gender) + C(Ownrent) + C(Selfempl) + C(edu_class)
'''
linear_model7 = ols(formula, train).fit()
linear_model7.summary()

# --- code cell ---
formula = '''
avg_exp_ln ~ dist_avg_income_ln + dist_home_val_ln + 
C(Selfempl) + C(gender) * C(edu_class) 
'''
linear_model8 = ols(formula, train).fit()
linear_model8.summary()

# ============================================================
# 保存结果
# ============================================================

# --- code cell ---
output_data_dir = './data_processed'
if not os.path.exists(output_data_dir):
    os.mkdir(output_data_dir)
output_data_path = os.path.join(output_data_dir, 
                                'creditcard_exp_linear_regression.csv')

# 训练集中的edu_class的取值包括“1， 2， 3”， 
# 而本期中的edu_class则只有“0， 1”，
# 因此将本期中的edu_class=0 概化为1

oos_data['edu_class'].replace({0: 1}, inplace=True)
oos_data['Pred_avg_exp'] = np.exp(linear_model8.predict(oos_data))
oos_data.to_csv(output_data_path, index=False)

# ============================================================
# # 4、正则算法
# ### 岭回归
# ============================================================

# --- code cell ---
# 对训练集的x进行标准化
continuous_xcols = ['Income', 'dist_home_val', 'dist_avg_income']
raw = train[continuous_xcols].copy()
train_scale = (raw - raw.mean()) / raw.std()

# 使用训练集均值与方差对测试集进行标准化！
oos_raw = oos_data[continuous_xcols].copy()
oos_scale = (oos_raw - raw.mean()) / raw.std()

# 将训练集的y添加进来
train_scale['avg_exp'] = train['avg_exp']

# 训练
linear_model_ridge = ols(
    'avg_exp ~ Income + dist_home_val + dist_avg_income',
    data=train_scale
).fit_regularized(
    alpha=1,    #  正则化系数的倒数
    L1_wt=0     #  `0`代表使用L2正则，即岭回归
)

# 预测
linear_model_ridge.predict(oos_scale)

# ============================================================
# ### 正则化系数选择：使用scikit-learn自动调优
# ============================================================

# --- code cell ---
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

#  抽取连续变量
continuous_xcols = ['Age', 'Income', 'dist_home_val', 'dist_avg_income']

# 标准化
scaler = StandardScaler()                           # ①模型初始化
scaler.fit(train[continuous_xcols])                 # ②使用fit训练模型
X = scaler.transform(train[continuous_xcols])       # ③使用transform转换
oos_scale = scaler.transform(
    oos_data[continuous_xcols])
y = train['avg_exp_ln']

# 参数搜索：空间 10^-2 到10^3的对数空间取100个值
alphas = np.logspace(-2, 3, 100, base=10)

# 交叉验证得到MSE最小的正则系数alphas
rcv = RidgeCV(
    alphas=alphas,      # 正则化系数的搜索空间
    cv=3,               # 3折交叉验证
)                                                   # ①初始化模型
rcv.fit(X, y)                                       # ②使用fit训练模型
rcv_predict = rcv.predict(oos_scale)                # ③使用predict预测
print("oos prediction:", np.exp(rcv_predict[:5]))   # 预测值取对数的逆变换

# 打印评估结果
print("coef of linear model:", rcv.coef_)
print('The best alpha is {}'.format(rcv.alpha_))
print('The r-square is {}'.format(rcv.score(X, y))) # Default: rsquared


# ============================================================
# ### 正则化系数选择：手动选择
# ============================================================

# ============================================================
# 岭迹图
# ============================================================

# --- code cell ---
from sklearn.linear_model import Ridge

#  使用不同的正则参数生成一系列岭回归模型
ridge = Ridge()
coefs = []
for alpha in alphas:
    ridge.set_params(alpha=alpha)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

#  绘制正则参数-回归系数的图形
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')          # 设置x轴为对数坐标
plt.xlabel('alpha')  
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# --- code cell ---
ridge.set_params(alpha=10)
ridge.fit(X, y)

print("Coefs of the model: ", ridge.coef_)
print("The r-square of the model:" , ridge.score(X, y))
print("Prediction of the model: ", np.exp(ridge.predict(oos_scale)[:5]))

# ============================================================
# ### LASSO算法
# ============================================================

# --- code cell ---
lmr1 = ols('avg_exp ~ Age + Income + dist_home_val + dist_avg_income',
           data=train).fit_regularized(alpha=1, L1_wt=1)

# ============================================================
# 正则化系数自动选择
# ============================================================

# --- code cell ---
from sklearn.linear_model import LassoCV

# 10折交叉验证获取最优正则系数
lasso_alphas = np.logspace(-3, 0, 100, base=10)
lcv = LassoCV(alphas=lasso_alphas, cv=10) # Search the min MSE by CV
lcv.fit(X, y)

print('The best alpha is {}'.format(lcv.alpha_))
print('The r-square is {}'.format(lcv.score(X, y)))  # Default score is rsquared

# ============================================================
# 正则化参数手动选择
# ============================================================

# --- code cell ---
from sklearn.linear_model import Lasso

# 选择不同参数生成lasso回归的系数
lasso = Lasso()
lasso_coefs = []
for alpha in lasso_alphas:
    lasso.set_params(alpha=alpha)
    lasso.fit(X, y)
    lasso_coefs.append(lasso.coef_)

# 绘制lasso正则系数图
ax = plt.gca()
ax.plot(lasso_alphas, lasso_coefs)
ax.set_xscale('log')    # x设为对数坐标
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# --- code cell ---
# 手动设置为0.02
lasso.set_params(alpha=0.02)
lasso.fit(X, y)

print("Coefs of the model: ", lasso.coef_)
print("The best r-square of the model:" , lasso.score(X, y))
print("Prediction of the model: ", np.exp(lasso.predict(oos_scale)[:5]))

# ============================================================
# #### *弹性网络
# ============================================================

# --- code cell ---
from sklearn.linear_model import ElasticNetCV

l1_ratio = [0.01, .1, .2, .4, .8, .95, .99, 1]

encv = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=100, n_jobs=-1, cv=4)
encv.fit(X, y)

# --- code cell ---
print('The best l1_ratio is {}'.format(encv.l1_ratio_))
print('The best alpha is {}'.format(encv.alpha_))
print('The best r-square of the model is : ', encv.score(X, y))
print('The prediction of the ElasticNet is : ', np.exp(encv.predict(oos_scale)[:5]))

# --- code cell ---


# --- code cell ---

