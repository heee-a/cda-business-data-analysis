"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
# 该银行交易行为数据有四个特征，分别表示去柜台服务、使用 Atm、刷 pos 机和信用卡的次数。
# 1、导入数据
import pandas as pd
model_data = pd.read_csv("profile_bank.csv")
data = model_data.loc[ :,'CNT_TBM':'CNT_CSC']
data.head()

# --- code cell ---
# - 2、查看相关系数矩阵，判定做变量降维的必要性（非必须）
corr_matrix = data.corr(method='pearson')
#corr_matrix = corr_matrix.abs()
corr_matrix

# --- code cell ---
# - 3、做主成分之前，进行中心标准化
from sklearn import preprocessing
data = preprocessing.scale(data)

# --- code cell ---
# - 4、使用 sklearn的主成分分析，用于判断保留主成分的数量
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
newData=pca.fit(data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

# --- code cell ---
# - 1、导入包，并对输入的数据进行主成分提取。为保险起见，data需要进行中心标准化
from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
fa = FactorAnalysis.load_data_samples(data,preproc_demean=True,preproc_scale=True)
fa.extract_components()

# --- code cell ---

# - 2、设定提取主成分的方式。默认为“broken_stick”方法，建议使用“top_n”法
fa.find_comps_to_retain(method='top_n',num_keep=3)

# --- code cell ---
# - 3、通过最大方差法进行因子旋转
fa.rotate_components(method='varimax')
fa_plotting.graph_summary(fa)

# --- code cell ---
# - 4、获取因子得分
pd.DataFrame(fa.comps["rot"])

# --- code cell ---
score=fa.get_component_scores(data)
score=pd.DataFrame(score)


# --- code cell ---

fa_scores=score.rename(columns={0: "ATM_POS", 1: "TBM",2: "CSC"})

# --- code cell ---
fa_scores.head()

# ============================================================
# - 识别异常
# ============================================================

# --- code cell ---
# - 1、查看变量的偏度
var = ["ATM_POS","TBM","CSC"]
skew_var = {}
for i in var:
    skew_var[i]=abs(fa_scores[i].skew())
skew=pd.Series(skew_var).sort_values(ascending=False)
skew


# --- code cell ---
# - 2、进行 k-means聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3) #MiniBatchKMeans()分批处理
#kmeans = cluster.KMeans(n_clusters=3, init='random', n_init=1)
result=kmeans.fit(fa_scores)
#print(result)

# --- code cell ---
# - 3、对分类结果进行解读
model_data_l=model_data.join(pd.DataFrame(result.
labels_))
model_data_l=model_data_l.rename(columns={0: "clustor"})
model_data_l.head()

# --- code cell ---
import matplotlib
get_ipython().magic("matplotlib inline")
model_data_l.clustor.value_counts().plot(kind = "pie")

# ============================================================
# - 用户细分
# ============================================================

# --- code cell ---
# - 1、进行变量分布的正态转换
import numpy as np
from sklearn import preprocessing

# --- code cell ---
quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
fa_scores_trans=quantile_transformer.fit_transform(fa_scores)
fa_scores_trans=pd.DataFrame(fa_scores_trans)

# --- code cell ---
fa_scores_trans=fa_scores_trans.rename(columns={0: "ATM_POS", 1: "TBM", 2: "CSC"})

# --- code cell ---
# 查看变量的偏度
var = ["ATM_POS","TBM","CSC"]
skew_var = {}
for i in var:
    skew_var[i]=abs(fa_scores_trans[i].skew())
skew=pd.Series(skew_var).sort_values(ascending=False)
skew


# --- code cell ---
# - 2、进行 k-means聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3) #MiniBatchKMeans()分批处理
#kmeans = cluster.KMeans(n_clusters=3, init='random', n_init=1)
result=kmeans.fit(fa_scores_trans)
#print(result)

# --- code cell ---
# - 3、对分类结果进行解读
model_data_l=model_data.join(pd.DataFrame(result.
labels_))
model_data_l=model_data_l.rename(columns={0: "clustor"})
model_data_l.head()

# --- code cell ---
model_data_l.clustor.value_counts()

# --- code cell ---
#聚类事后分析
from sklearn import tree
clf=tree.DecisionTreeClassifier(criterion='gini',max_depth=3, min_samples_split=100, min_samples_leaf=100,random_state=12345)
# 当前支持计算信息增益和 GINI
clf.fit(model_data, result.labels_)

# --- code cell ---
import pydotplus
from IPython.display import Image
import sklearn.tree as tree
dot_data = tree.export_graphviz(clf,out_file=None,feature_names=model_data.columns,class_names=["0","1","2"],filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# --- code cell ---

