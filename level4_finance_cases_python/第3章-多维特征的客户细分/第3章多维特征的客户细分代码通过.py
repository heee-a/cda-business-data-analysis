"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 第三章多维特征的客户细分代码
# ============================================================

# ============================================================
# # 3.1客户细分数据说明
# ============================================================

# ============================================================
# 数据读取
# ============================================================

# --- code cell ---
# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",7)  # 为方便截图设置最大显示列数为7
bank = pd.read_csv('data_clust.csv',skipinitialspace=True)
bank.describe(include='all')

# ============================================================
# # 3.2 预处理
# ============================================================

# ============================================================
# 3.2.1 填补缺失值
# ============================================================

# --- code cell ---
for col in bank.columns:
    if bank[col].hasnans:
        print(col)

# ============================================================
# 这四个有缺失值的字段中，interested_travel、computer_owner、HH_adults_num这三个字段是数值型分类变量，而且缺失比例并不大，因此用它们各自的众数进行填补
# ============================================================

# --- code cell ---
fill_cols = ['interested_travel', 'computer_owner', 'HH_adults_num']
fill_values = {col: bank[col].mode()[0] for col in fill_cols}
bank = bank.fillna(fill_values)

# ============================================================
# 3.2.2 修订错误值
# ============================================================

# --- code cell ---
bank['HH_has_children'] = bank['HH_has_children']\
    .replace({'N':0, 'Y':1, np.NaN:0})

# --- code cell ---
bank['interested_reading'] = bank['interested_reading']\
    .replace({'.':'0'}).astype('int')

# ============================================================
# 3.2.3 处理离散型变量
# ============================================================

# --- code cell ---
from itertools import combinations
from scipy import stats
_cols = [
    'interested_travel',
    'computer_owner',
    'marital',
    'interested_golf', 
    'interested_gambling', 
    'HH_has_children',
    'interested_reading'
]
sample = bank[_cols].sample(3000, random_state=12345)

# 对离散变量两两进行卡方检验
for colA, colB in combinations(_cols, 2):
    crosstab = pd.crosstab(sample[colA], sample[colB])
    pval = stats.chi2_contingency(crosstab)[1]

# --- code cell ---
# 设定置信度阈值为0.05
if pval > 0.05:
        print('p-value = %0.3f between "%s" and "%s"' %(pval, colA, colB))


# --- code cell ---
#对interested_reading进行二值化
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=1.5)
bank['interested_reading'] = binarizer.fit_transform(
    bank[['interested_reading']])

# 生成偏好填充率
interest =[
    'interested_travel',
    'computer_owner',
    'interested_golf', 
    'interested_gambling',
    'interested_reading'
]
n_ = len(interest)

bank = bank.drop(interest, axis=1)\
               .assign(interest=bank[interest].sum(axis=1) / n_)
bank.head()

# ============================================================
# 3.2.4 正态化与标准化
# ============================================================

# --- code cell ---
continuous_cols = ['age', 'home_value', 'risk_score', 'interested_sport', 
                   'HH_dieting', 'auto_member', 'HH_grandparent',
                   'HH_head_age', 'loan_ratio']

categorical_cols = ['marital', 'interest', 'HH_adults_num']

discreate_cols = ['HH_has_children'] # 不进入聚类模型

# --- code cell ---
bank[continuous_cols].hist(bins=25,figsize=[8, 8])
plt.show()

# --- code cell ---
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(n_quantiles=100, output_distribution='normal')
qt_data = qt.fit_transform(bank[continuous_cols])

pd.DataFrame(qt_data,columns=continuous_cols).hist(bins=25, figsize=[8,8])
plt.show()

# --- code cell ---
from sklearn.preprocessing import scale

scale_data = scale(bank[categorical_cols])
# 合并数据
data = np.hstack([qt_data, scale_data, bank[discreate_cols]])
data = pd.DataFrame(
    data, columns=continuous_cols + categorical_cols + discreate_cols)
data.head()

# ============================================================
# # 3.3 维度分析
# ============================================================

# --- code cell ---
# 根据业务解释，将变量划分为大的维度 `household`, `hobby`
from sklearn.decomposition import PCA
household = ['age', 'marital', 'HH_adults_num', 'home_value', 
             'risk_score', 'HH_grandparent', 'HH_head_age', 'loan_ratio']
hobby = ['HH_dieting', 'auto_member','interest', 'interested_sport']

#  确定`household`维度中主成分解释的方差比率
pca_hh = PCA().fit(data[household])
pca_hh.explained_variance_ratio_.cumsum()

# --- code cell ---
from sklearn import preprocessing
data_household = preprocessing.scale(data[household])
data_hobby = preprocessing.scale(data[hobby])

# --- code cell ---
from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
fa_hh = FactorAnalysis.load_data_samples(
    data_household,
    preproc_demean=True,
    preproc_scale=True
)
fa_hh.extract_components()

# 使用“top_n”法保留4个因子
fa_hh.find_comps_to_retain(method='top_n',num_keep=4)

# 最大方差旋转
fa_hh.rotate_components(method='varimax')
fa_plotting.graph_summary(fa_hh)

# --- code cell ---
pd.DataFrame(fa_hh.comps['rot'].T, columns=household)

# ============================================================
# 对客户的“偏好属性”进行因子分析（保留3个因子）
# ============================================================

# --- code cell ---
fa_hb = FactorAnalysis.load_data_samples(
    data_hobby, 
    preproc_demean=True,
    preproc_scale=True
)
fa_hb.extract_components()
fa_hb.find_comps_to_retain(method='top_n', num_keep=3)
fa_hb.rotate_components(method='varimax')
pd.DataFrame(fa_hb.comps['rot'].T, columns=hobby)

# ============================================================
# 根据业务解释为每个因子赋予一个名字，得到的因子得分将作为聚类模型的输入
# ============================================================

# --- code cell ---
# 计算household因子得分
data_hh = pd.DataFrame(
    np.dot(data[household], fa_hh.comps['rot']), 
    columns=['life_circle','finance', 'HH_size', 'risk']
)

# 计算hobby因子得分
data_hb = pd.DataFrame(
    np.dot(data[hobby], fa_hb.comps['rot']), 
    columns=['sports', 'health', 'leisure']
)

# ============================================================
# # 3.4聚类
# ============================================================

# --- code cell ---
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster_plot(data, k_range=range(2, 12), n_init=5, sample_size=2000
                ):
    scores = []  # 保存模型效果的评分
    models = {}  # 保存模型

    # 遍历选择的所有k
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=n_init) # 初始化模型
        kmeans.fit(data)  # 训练
        models[k] = kmeans
        sil = silhouette_score(data, kmeans.labels_, 
                               sample_size=sample_size)
        scores.append([k, kmeans.inertia_, sil]) # inertia_即SSD 

    scores_df = pd.DataFrame(scores, columns=['k','sum_square_dist', 'sil'])

    #  绘制簇内离差平方和图
    plt.figure(figsize=[9, 2])
    plt.subplot(121, ylabel='sum_square')
    plt.plot(scores_df.k, scores_df.sum_square_dist)
   
#  绘制轮廓系数图
    plt.subplot(122, ylabel='silhouette_score')
    plt.plot(scores_df.k, scores_df.sil)
    plt.show()

    return models  # 返回不同k值对应的模型

# --- code cell ---
# 对因子得分进行标准化
from sklearn.preprocessing import StandardScaler
hh_scaler = StandardScaler()
scale_data_hh = hh_scaler.fit_transform(data_hh)
hb_scaler = StandardScaler()
scale_data_hb = hb_scaler.fit_transform(data_hb)

# 绘制簇内离差平方和、轮廓系数图
models_hh = cluster_plot(scale_data_hh)
print("-" * 80)
models_hb = cluster_plot(scale_data_hb)

# --- code cell ---
hh_labels = pd.DataFrame(models_hh[3].labels_, columns=['hh'])
hb_labels = pd.DataFrame(models_hb[2].labels_, columns=['hb'])
clusters = bank.join(hh_labels).join(hb_labels)
clusters.head()

# ============================================================
# # 3.5 簇特征的解释
# ============================================================

# --- code cell ---
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from IPython.display import Image
import sklearn.tree as tree

# 建立决策树
tree_hh = DecisionTreeClassifier()
tree_hb = DecisionTreeClassifier()
tree_hh.fit(clusters[household], clusters['hh'])
tree_hb.fit(clusters[hobby], clusters['hb'])

# 对household的聚类标签绘制决策树
dot_hh = tree.export_graphviz(
    tree_hh,
    out_file=None, 
    feature_names=household, 
    class_names=True,
    max_depth=2, 
    filled=True
) 
graph_hh = pydotplus.graph_from_dot_data(dot_hh)  
Image(graph_hh.create_png())

# --- code cell ---
dot_hb = tree.export_graphviz(
    tree_hb,
    out_file=None, 
    feature_names=hobby, 
    class_names=True,
    max_depth=2, 
    filled=True
) 

graph_hb = pydotplus.graph_from_dot_data(dot_hb)  
Image(graph_hb.create_png())

# --- code cell ---
cluster_pt = pd.pivot_table(clusters, index='hh', columns='hb',
                            aggfunc='mean').T
cluster_pt.swaplevel('hb', 0).sort_index(level=0)

# --- code cell ---


# --- code cell ---

