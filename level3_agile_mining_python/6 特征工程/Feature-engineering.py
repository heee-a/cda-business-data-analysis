"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # Preprocessing
# ============================================================

# --- code cell ---
# %matplotlib inline

import pandas as pd

pd.set_option('display.max_columns', 10)

# --- code cell ---
raw = pd.read_csv('telecom_bill.csv')
raw.head()

# ============================================================
# ### Wrong Value Processing
# ============================================================

# --- code cell ---
raw[raw.traffic >= 5000].head()

# --- code cell ---
data = raw[raw.traffic < 5000].copy()

# --- code cell ---
statesCount = data['ID'].groupby(data.state).count()
statesCount.head(10)

# --- code cell ---
import matplotlib.pyplot as plt

plt.pie(statesCount.values, labels = statesCount.index,
        colors=('b','g','r','c','m','y','tan','w'), autopct='%1.1f%%')

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')  
plt.show()

# --- code cell ---
import numpy as np

new_state = data['state'].replace([2, 4], np.nan)

# ============================================================
# ### Outlier Processing
# ============================================================

# --- code cell ---
plt.boxplot(data.expenditure)
plt.show()

# --- code cell ---
tmp = data.expenditure.copy()

# Compute boundray.
mu = data.expenditure.mean()
sigma = data.expenditure.std()
lb = mu - 3 * sigma
hb = mu + 3 * sigma

# Outier processing
tmp[tmp < lb] = lb
tmp[tmp > hb] = hb

plt.boxplot(tmp)
plt.show()

# --- code cell ---
data.expenditure.hist(figsize=(8, 4), bins=40)
plt.show()

# --- code cell ---
data['log_expd'] = np.log1p(tmp)  # 等同于np.log(tmp + 1)
data.log_expd.hist(figsize=(8, 4), bins=20)
plt.show()

# --- code cell ---
from sklearn.preprocessing import FunctionTransformer

functionTF = FunctionTransformer(np.log1p)
log_data = functionTF.fit_transform(data[['call', 'traffic']])
data['log_call'] = log_data.iloc[:, 0]
data['log_traffic'] = log_data.iloc[:, 1]
data.head()

# ============================================================
# ### Missing Data Imputation
# ============================================================

# --- code cell ---
data.count()

# --- code cell ---
data.isnull().sum()

# --- code cell ---
data.dropna(thresh=1, subset=['state', 'IMEI']).count()

# --- code cell ---
# 将所有缺失值补位 0，但不替换。不适用于一般情况
data.fillna(value=0, inplace=False).head()

# --- code cell ---
top_state = data['state'].mode()[0]
data.fillna({"IMEI": 0, "state": top_state}, inplace=True)
data.count()

# --- code cell ---
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit_transform(raw[['state', 'IMEI']])

# ============================================================
# ### Binarization
# ============================================================

# --- code cell ---
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=0, copy=True)
data['has_IMEI'] = binarizer.fit_transform(data[['IMEI',]])
data.head()

# --- code cell ---
pd.cut(data['IMEI'], bins=[-float('inf'), 0, float('inf')], 
       labels=False).head()

# ============================================================
# ### Discretization
# ============================================================

# ============================================================
# 等宽离散
# ============================================================

# --- code cell ---
from datetime import datetime

dura_col = datetime(2015, 1, 1) - pd.to_datetime(data['join_time'])
data['duration'] = dura_col.map(lambda x: x.days / 30.)

# --- code cell ---
equal_width = pd.cut(data['duration'], 5)
data['churn'].groupby(equal_width).agg(['count', 'mean','sum'])

# --- code cell ---
sample_labels, retbins = pd.cut(data['duration'], 5, labels=range(5),
                                retbins=True)
print(sample_labels[:5])
print(retbins)

# ============================================================
# 等分离散
# - 老版本的pandas的qcut方法有bug，现有修复
# ============================================================

# --- code cell ---
equal_depth = pd.qcut(data['duration'], 14)
discretization_dura = data['churn'].groupby(equal_depth).agg(
    ['count', 'mean','sum'])
discretization_dura.head()

# --- code cell ---
discretization_dura['mean'].plot(kind='bar')
plt.show()

# ============================================================
# 人工离散
# ============================================================

# --- code cell ---
bins = [-np.inf, 5, 12, 36, np.inf]

artificial_width = pd.cut(data['duration'], bins)
data['churn'].groupby(artificial_width).agg(['mean']).plot(kind='bar')
plt.show()

# ============================================================
# ### Dummy coding
# ============================================================

# --- code cell ---
dummies = pd.get_dummies(data.state, prefix='s')
data.join(dummies).head()

# --- code cell ---
from sklearn.preprocessing import OneHotEncoder

OneHotEncoder().fit_transform(data[['state',]]).toarray()

# ============================================================
# ### Standardization
# ============================================================

# --- code cell ---
from sklearn.preprocessing import StandardScaler

consumptions = data.loc[:, 'log_expd':'log_traffic']
std_scaler = StandardScaler()
std_data = std_scaler.fit_transform(consumptions)

# --- code cell ---
std_df = pd.DataFrame(
    std_data, 
    columns=['s_expd','s_call','s_traffic'], 
    index=data.index
)

data1 = data.join(std_df)
data1.head()

# --- code cell ---
from sklearn.preprocessing import MinMaxScaler

range_scaler = MinMaxScaler()
range_scaler.fit_transform(consumptions)[:5]

# ============================================================
# ### Normalization
# ============================================================

# --- code cell ---
from sklearn.preprocessing import Normalizer

Normalizer().fit_transform(data1.iloc[:, -3:])

# ============================================================
# # Feature Construction
# ============================================================

# --- code cell ---
# Duration has been constructed before
data1['unit_price'] = data1['expenditure'] / (data1['traffic'] + 1.)
data1['TAC'] = data1['IMEI'].map(lambda x: str(x)[:6])

data1.head()

# ============================================================
# # Feature Extraction
# ============================================================

# ============================================================
# ### PCA
# ============================================================

# --- code cell ---
from sklearn.decomposition import PCA

pca = PCA(n_components=2, whiten=True)
pca.fit_transform(data1[['s_expd','s_call', 's_traffic']])

# --- code cell ---
pca.explained_variance_ratio_ # 每个主成分能解释的变异值的百分比

# --- code cell ---
pca.explained_variance_  # 每个主成分能解释的变异值

# --- code cell ---
pca.components_  # 主成分方向

# --- code cell ---
# 手动构造一个伪测试集，并投影到训练集构造的主成分方向上
pca.transform(
    pd.DataFrame( 
        data=[[-1, 0, 1], [1, 2, 3]], 
        columns=['s_expd','s_call', 's_traffic'])
)

# ============================================================
# ### LDA
# ============================================================

# --- code cell ---
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit_transform(data1[['s_expd', 's_call', 's_traffic']], data1.churn)

# --- code cell ---
lda.coef_  # 投影的方向

# --- code cell ---
# 对人工造的测试集进行特征转换（投影）
lda.transform(
    pd.DataFrame(
        [[-1, 0, 1],[1, 2, 3]],
        columns=['s_expd', 's_call', 's_traffic']
    )
)

# --- code cell ---
# 线性判别分析可以直接作为分类器模型
prediction = lda.predict(data1[['s_expd','s_call','s_traffic']])
result = prediction == data1.churn
result.mean()  # 预测的准确率

# ============================================================
# # Feature Selection
# ============================================================

# ============================================================
# ### Filter
# ============================================================

# ============================================================
# 方差过滤
# ============================================================

# --- code cell ---
from sklearn.feature_selection import VarianceThreshold

var_th = VarianceThreshold(threshold=0.09)
var_th.fit_transform(data1[['vip', 'has_IMEI']])

# --- code cell ---
# 各个变量的方差
var_th.variances_

# ============================================================
# 卡方检验
# ============================================================

# --- code cell ---
# 将待选特征与目标变量按照类型进行划分
x_categorical = data1[['vip', 'state', 'has_IMEI']]
x_continuous = data1[['s_expd', 's_call', 's_traffic', 'duration']]
y_categorical = data1['churn']
y_continuous = data1['s_expd']

# --- code cell ---
from sklearn.feature_selection import chi2

# 两个离散变量的相关性可以使用卡方检验
chi2(x_categorical, y_categorical) 

# --- code cell ---
from sklearn.feature_selection import SelectKBest

# 通过卡方检验保留与目标变量最相关的两个变量
selector = SelectKBest(chi2, k=2).fit(x_categorical, y_categorical)
selector.transform(x_categorical)

# --- code cell ---
# 查看被筛选掉的变量

selector.get_support()

# ============================================================
# ANOVA
# ============================================================

# --- code cell ---
from sklearn.feature_selection import f_classif

# ANOVA用于判断连续变量和离散变量的相关程度
f_classif(x_continuous, y_categorical)

# --- code cell ---
# 强行保留与目标变量最相关的两个连续变量
f_classif_selector = SelectKBest(f_classif, k=2)
print(f_classif_selector.fit_transform(x_continuous, y_categorical))
# f_classif_selector.get_support()  # 输出被筛选掉的变量

# --- code cell ---
f_classif_selector.get_support()

# ============================================================
# 相关系数
# ============================================================

# --- code cell ---
from sklearn.feature_selection import f_regression

# 相关系数用于分析连续变量之间的相关性
f_regression(x_continuous, y_continuous)

# --- code cell ---
# 保留与消费金额最相关的两个变量
f_regression_selector = SelectKBest(f_regression, k=2)
print(f_regression_selector.fit_transform(x_continuous, y_continuous))


# --- code cell ---
# 查看被筛选掉的变量
f_regression_selector.get_support() 

# ============================================================
# ### Wrapper
# ============================================================

# --- code cell ---
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# RFE是Recursive Feature Elimination（递归特征消除）的缩写
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2) # 保留两个特征
rfe.fit_transform(x_continuous, y_categorical)

# --- code cell ---
rfe.get_support()

# ============================================================
# ### Embedded
# ============================================================

# --- code cell ---
from sklearn.ensemble import RandomForestClassifier

# 很多模型天然具有生成变量重要性的能力，例如随机森林
rfc = RandomForestClassifier()
rfc.fit(x_continuous, y_categorical)

list(zip(x_continuous.columns, rfc.feature_importances_))

# --- code cell ---
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

# GBDT模型也具有生成变量重要性的能力
gbdt_selector = SelectFromModel(GradientBoostingClassifier(),
                                threshold='mean')
gbdt_selector.fit_transform(x_continuous, y_categorical)

# --- code cell ---
gbdt_selector.get_support()

# --- code cell ---


# --- code cell ---

