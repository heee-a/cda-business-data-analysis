"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 主成分分析
# ============================================================

# ============================================================
# ```
# X1	GDP
# X2	人均GDP
# X3	工业增加值
# X4	第三产业增加值
# X5	固定资产投资
# X6	基本建设投资
# X7	社会消费品零售总额
# X8	海关出口总额
# X9	地方财政收入
# ```
# ============================================================

# ============================================================
# ## 导入数据
# ============================================================

# --- code cell ---
import pandas as pd

# --- code cell ---
#1.导入数据
model_data = pd.read_csv("cities_10.csv",encoding='gbk')
model_data.head()

# --- code cell ---
data = model_data.loc[ :,'X1':]
data.head()

# ============================================================
# ## 中心标准化
# ============================================================

# --- code cell ---
# 3.做主成分之前，进行中心标准化
from sklearn import preprocessing
data = preprocessing.scale(data)

# --- code cell ---
data

# ============================================================
# ## 计算各主成分解释的变异
# ============================================================

# --- code cell ---
# 4.使用sklearn的主成分分析，用于判断保留主成分的数量
from sklearn.decomposition import PCA

# ============================================================
# ``` 
# 说明：1、第一次的n_components参数应该设的大一点
# 说明：2、观察explained_variance_ratio_和explained_variance_的取值变化，建议explained_variance_ratio_累积大于0.85,explained_variance_需要保留的最后一个主成分大于0.8，
# ```
# ============================================================

# --- code cell ---
pca = PCA(n_components=9)
pca.fit(data)

# --- code cell ---
print(pca.explained_variance_)#解释的方差（对于标准化数据来说，单个主成分解释的方差应该大于1）
print(pca.explained_variance_ratio_)

# ============================================================
# ## 选取主成分保留个数
# ============================================================

# --- code cell ---
pca = PCA(n_components=2).fit(data)

# --- code cell ---
#特征值
pca.explained_variance_

# ============================================================
# ## 计算主成分得分
# ============================================================

# ============================================================
# # 因子分析
# ============================================================

# ============================================================
# 安装fa_kit
# ============================================================

# ============================================================
# ## 导入数据
# ============================================================

# --- code cell ---
# 1.因子分析的概念很多，作为刚入门的人，我们可以认为因子分析是主成分分析的延续
from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting

# --- code cell ---
fa = FactorAnalysis.load_data_samples(
        data,     #标准化以后的数据
        preproc_demean=True,
        preproc_scale=True
        ) 
fa.extract_components()  

# ============================================================
# ## 主成分法
# ============================================================

# --- code cell ---
# 2.设定提取主成分的方式。默认为“broken_stick”方法，建议使用“top_n”法
fa.find_comps_to_retain(method='top_n',num_keep=2)

# ============================================================
# ## 最大方差旋转法
# ============================================================

# --- code cell ---
# 3.通过最大方差法进行因子旋转
fa.rotate_components(method='varimax')
pd.DataFrame(fa.comps["rot"])#查看因子权重

# ============================================================
# ## 计算因子得分
# ============================================================

# --- code cell ---
# 4.获取因子得分
#以下是矩阵相乘的方式计算因子：因子=原始数据（n*k）*权重矩阵(k*num_keep)  

# --- code cell ---
import numpy as np

# --- code cell ---
fas = pd.DataFrame(fa.comps["rot"])   #因子权重
data = pd.DataFrame(data)#注意data数据需要标准化
fa_score = pd.DataFrame(np.dot(data, fas))

# --- code cell ---
data

# --- code cell ---
fas

# --- code cell ---
#因子得分
fa_score

# --- code cell ---
#因子命名
a=fa_score.rename(columns={0: "Gross", 1: "Avg"})
citi10_fa=model_data.join(a) 

# --- code cell ---
citi10_fa

# --- code cell ---
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# --- code cell ---
import matplotlib.pyplot as plt
x=citi10_fa['Gross']
y=citi10_fa['Avg']
label=citi10_fa['AREA']
plt.scatter(x, y)
for a,b,l in zip(x,y,label):
    plt.text(a, b+0.1, '%s.' % l, ha='center', va= 'bottom',fontsize=14)

plt.show()

# --- code cell ---


# --- code cell ---

