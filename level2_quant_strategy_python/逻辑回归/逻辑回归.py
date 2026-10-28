"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# 变量说明：
# - subscriberID="个人客户的ID"
# - churn="是否流失：1=流失";
# - gender="性别"
# - Age="年龄"
# - edu_class="教育水平"
# - incomeCode="用户居住区域平均收入的代码"
# - duration="在网时长"
# - feton="是否注册飞信"
# - peakMinAv="统计期间内最高单月通话时长"
# - peakMinDiff="统计期间结束月份与开始月份相比通话时长增加数量"
# - posTrend="该用户通话时长是否呈现出上升态势：是=1"
# - negTrend="该用户通话时长是否呈现出下降态势：是=1"
# - nrProm="电话公司营销的数量"
# - prom="最近一个月是否被营销过：是=1"
# - curPlan="统计时间开始时套餐类型：1=最高通过200分钟；2=300分钟；3=350分钟；4=500分钟"
# - avgplan="统计期间内平均套餐类型"
# - planChange="统计期间是否更换过套餐：1=是"
# - posPlanChange="统计期间是否提高套餐：1=是"
# - negPlanChange="统计期间是否降低套餐：1=是"
# - call_10086="拨打10086的次数"
# ============================================================

# --- code cell ---
import os
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ============================================================
# # 导入数据
# ============================================================

# --- code cell ---
churn = pd.read_csv('telecom_churn.csv')   
churn.head()

# --- code cell ---
churn.info()

# ============================================================
# # 分类变量的相关关系
# ============================================================

# ============================================================
# ## 交叉表
# ============================================================

# --- code cell ---
# normalize=False默认计算频数
cross_table = pd.crosstab(churn.posTrend, churn.churn, margins=True)
cross_table

# --- code cell ---
def percConvert(ser):
    return ser/float(ser[-1])

cross_table.apply(percConvert, axis = 1)

# --- code cell ---
544/1644

# ============================================================
# ## 卡方检验
# ============================================================

# --- code cell ---
#列联表的卡方独立性检验：显著，说明二者存在相关关系
print('''chisq = %6.4f 
p-value = %6.4f
dof = %i 
expected_freq = %s'''  %stats.chi2_contingency(cross_table.iloc[:2, :2]))

# ============================================================
# # 一元逻辑回归
# ============================================================

# ============================================================
# ## 划分训练集与测试集
# ============================================================

# --- code cell ---
#在网时长与是否流失
churn.plot(x='duration', y='churn', kind='scatter');

# --- code cell ---
# 随机抽样，建立训练集与测试集
train = churn.sample(frac=0.7, random_state=666)
test = churn[~ churn.index.isin(train.index)]
print(' 训练集样本量: %i \n 测试集样本量: %i' %(len(train), len(test)))

# ============================================================
# ## 建立模型
# ============================================================

# --- code cell ---
#建模
lg = smf.logit('churn ~ duration', train).fit()
lg.summary()

# ============================================================
# ## 模型预测
# ============================================================

# --- code cell ---
#预测（概率）
train['proba'] = lg.predict(train)
test['proba'] = lg.predict(test)

# --- code cell ---
test[['churn','proba']].head()

# ============================================================
# ## 模型评估
# ============================================================

# --- code cell ---
import sklearn.metrics as metrics

# --- code cell ---
#设置阈值
test['prediction'] = (test['proba'] > 0.3).astype('int')

# --- code cell ---
test[['churn','prediction']].head(10)

# --- code cell ---
#混淆矩阵
pd.crosstab(test.churn, test.prediction, margins=True)

# --- code cell ---
#计算准确率
acc = sum(test['prediction'] == test['churn']) /len(test)
print('The accurancy is %.2f' %acc)

# --- code cell ---
#ROC曲线
for i in np.arange(0.1, 0.9, 0.1):
    prediction = (test['proba'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction,test.churn,
                                   margins = True)
    precision = confusion_matrix.loc[0, 0] /confusion_matrix.loc['All', 0]
    recall = confusion_matrix.loc[0, 0] / confusion_matrix.loc[0, 'All']
    Specificity = confusion_matrix.loc[1, 1] /confusion_matrix.loc[1,'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold: %s, precision: %.2f, recall:%.2f ,Specificity:%.2f , f1_score:%.2f'%(i, precision, recall, Specificity,f1_score))

# --- code cell ---
fpr_test, tpr_test, th_test = metrics.roc_curve(test.churn, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.churn, train.proba)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()

# --- code cell ---
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

# --- code cell ---
#最优阈值（约登指数）
th_test[(tpr_test - fpr_test).argmax()]

# ============================================================
# # 多元逻辑回归
# ============================================================

# ============================================================
# ## 特征选择
# ============================================================

# ============================================================
# ### 连续变量选择
# ============================================================

# ============================================================
# 前向回归、后向回归、逐步回归：
# - 前向回归：Y和每个X做回归，选择解释力度最高的（R方、p_value、AIC、BIC），第一个变量进入回归方程，进行F检验和t检验，同时计算残差平方和s1，通过检验则保留；引入第二个变量，重新构建方程与检验，计算残差平方和s2，比较s1与s2，s1-s2（偏回归平方和）
# - 后向回归：与前向回归相反，所有变量一次进入，进行检验，根据偏回归平方和逐个剔除
# - 逐步回归：结合两者，变量一个个进入，引入新变量要进行检验，通过才加入，引入新变量后要对原变量重新进行检验，一旦不显著则剔除
# ============================================================

# --- code cell ---
# model = smf.glm(
#         formula=formula, data=data, 
#         family=sm.families.Binomial(sm.families.links.logit)
#     ).fit()

# --- code cell ---
#向前法
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
            aic = smf.logit(
                formula=formula, data=data).fit().aic
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
    model = smf.logit(formula=formula, data=data).fit()
    return(model)

# --- code cell ---
#只有连续变量可以通过逐步回归进行筛选
candidates = ['churn','duration','AGE','incomeCode','nrProm','peakMinAv','peakMinDiff','call_10086']
data_for_select = train[candidates]

# --- code cell ---
lg_m1 = forward_select(data=data_for_select, response='churn')
lg_m1.summary()

# --- code cell ---
print(f'原来有 {len(candidates)-1} 个变量')
print(f'筛选剩下 {len(lg_m1.params.index.tolist())} 个(包含 intercept 截距项)。')

# ============================================================
# ### 分类变量选择
# ============================================================

# ============================================================
# 解决方案：   
# - 逐一根据显著性测试   
# - 使用sklearn中的决策树等方法筛选变量，但是多分类变量需要事先进行变量概化   
# - 使用WOE转换，多分类变量也需要事先进行概化，使用scorecardpy包中的woe算法可以自动进行概化   
# ============================================================

# --- code cell ---
class_col = ['posTrend','prom','edu_class','feton','curPlan','avgplan','gender','negTrend','planChange','posPlanChange','negPlanChange']
for i in class_col:
    tab = pd.crosstab(churn[i], churn.churn)
    print(i,''' p-value = %6.4f'''  %stats.chi2_contingency(tab)[1])

# ============================================================
# ## 模型建立
# ============================================================

# --- code cell ---
formula = '''churn ~ duration + AGE +  incomeCode + peakMinDiff + call_10086 +C(edu_class) +C(feton)+ C(posTrend) + C(negTrend) + C(curPlan) + C(avgplan) + C(gender)'''
lg_m = smf.logit(formula=formula, data=train).fit()
lg_m.summary()

# ============================================================
# ## 模型预测
# ============================================================

# --- code cell ---
train['proba'] = lg_m.predict(train)
test['proba'] = lg_m.predict(test)

# --- code cell ---
test[['churn','proba']].head(10)

# ============================================================
# ## 模型评估
# ============================================================

# --- code cell ---
#混淆矩阵（阈值）
test['prediction'] = (test['proba'] > 0.5).astype('int')
pd.crosstab(test.churn, test.prediction, margins=True)

# --- code cell ---
#准确率
acc_m = sum(test['prediction'] == test['churn']) /len(test)
print('The accurancy is %.2f' %acc_m)

# --- code cell ---
#ROC曲线
for i in np.arange(0.1, 0.9, 0.1):
    prediction = (test['proba'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction,test.churn,
                                   margins = True)
    precision = confusion_matrix.loc[0, 0] /confusion_matrix.loc['All', 0]
    recall = confusion_matrix.loc[0, 0] / confusion_matrix.loc[0, 'All']
    Specificity = confusion_matrix.loc[1, 1] /confusion_matrix.loc[1,'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold: %s, precision: %.2f, recall:%.2f ,Specificity:%.2f , f1_score:%.2f'%(i, precision, recall, Specificity,f1_score))

# --- code cell ---
fpr_test, tpr_test, th_test = metrics.roc_curve(test.churn, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.churn, train.proba)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()

# --- code cell ---
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

# --- code cell ---
#最优阈值
th_test[(tpr_test - fpr_test).argmax()]

# --- code cell ---
test['prediction'] = (test['proba'] > 0.456).astype('int')
pd.crosstab(test.churn, test.prediction, margins=True)

# ============================================================
# ## 模型优化
# ============================================================

# --- code cell ---
# 方差膨胀因子检测
def vif(df, col_i):
    from statsmodels.formula.api import ols
    
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)

# --- code cell ---
exog = train[candidates].drop(['churn'], axis=1)
for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i=i))

# --- code cell ---
#分类变量的相关性检验
#最终模型的显著性检验

# --- code cell ---

