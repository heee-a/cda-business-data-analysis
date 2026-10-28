"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
# !pip install shap

# ============================================================
# 机器学习模型的可解释性是指人类能够理解或推断模型决策或预测的能力。这是机器学习领域中的一个重要研究方向，尤其在高风险和高影响力的应用场景（如医疗、生产和司法）中，模型可解释性对信任建立、问题诊断和合规性至关重要。
# ============================================================

# ============================================================
# 可解释性的实际应用举例：
# - 医疗领域：解释模型为什么预测某患者患某种疾病，以辅助医生决策。
# - 金融领域：解释贷款审批或信用评分的依据，确保公平性和合规性。
# - 工业制造：分析预测设备故障的特征，以优化维护计划。
# ============================================================

# --- code cell ---
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection  import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
Xtrain, Xtest, Ytrain, Ytest= train_test_split(data, labels, test_size= 0.3, stratify= labels, random_state= 42)

# ============================================================
# # 简单的模型解释（白盒模型）
# 白盒模型指的是模型的所有行为都可以通过解释指标被人所理解，例如 回归模型，KNN，朴素贝叶斯 等都是常用的白盒模型。
#
# 这里以回归模型为例，可以看到模型的所有常用解释指标都可以通过统计学方法直接计算得出。
# ============================================================

# --- code cell ---
import statsmodels.api as sm
Xtrain_lr= Xtrain.copy()
Xtrain_lr["const"]= 1

LR= sm.Logit(Ytrain, Xtrain_lr).fit()
LR.summary()

# ============================================================
# # 进阶的模型解释方法
# ============================================================

# ============================================================
# 对于一些更加复杂的模型，例如树模型和神经网络模型，有时无法直接通过模型的统计指标来解释模型的行为。
#
# 此时我们可以引入如下的分析方法和相应的工具：
# - 特征重要性分析
# - 特征影响量化分析
# - 决策路径分析
# - 规则提取
# - 反事实解释（Counterfactual Explanation）
# ============================================================

# ============================================================
# # 特征重要性分析
# 特征重要性适用于全局分析，强调模型整体的特征贡献，是优化特征工程、简化数据集的首选。可以帮助我们更好地理解数据，降低模型复杂度。还可以指导指导业务人员优先关注这些重要因素。
# ============================================================

# ============================================================
# ## 模型输出的特征重要性
# ============================================================

# ============================================================
# ### 决策树
# ============================================================

# ============================================================
# - 遍历所有使用过该特征的分裂节点：
# 统计该特征在所有分裂节点上产生的不纯度减少量（信息增益）。
#
# - 对每个特征的重要性求和：
# 将该特征在树中所有节点上的不纯度减少量（信息增益）累加，得到该特征的重要性。
#
# - 归一化处理：
# 将所有特征的重要性值归一化，使它们的总和等于 1
#
# 缺点：所有基于决策树的模型，feature_importances_ 会偏向于高基数（具有更多分裂点）的特征和高方差的特征。
# ============================================================

# --- code cell ---
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(Xtrain, Ytrain)

dt.feature_importances_

# ============================================================
# ### 随机森林
# ============================================================

# ============================================================
# - 首先对每棵树计算特征重要性。
# - 最终的 feature_importances_ 是所有树中特征重要性的均值（即对每棵树的特征重要性求和后取平均）。
# - 若启用OOB。在计算特定特征的重要性时，随机扰动某个特征，并观察模型对 OOB 数据的预测性能下降程度。如果扰动该特征导致性能下降较多，则说明该特征对模型的贡献较大，其重要性也更高。OOB 数据是独立于训练过程的，因此可以更好地反映特征对模型实际性能的影响。这种计算方式更加稳健，能有效避免过拟合。
#
# ============================================================

# --- code cell ---
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(oob_score= True, n_jobs= -1)
rf.fit(Xtrain, Ytrain)

rf.feature_importances_

# ============================================================
# ### XGBoost
# ============================================================

# ============================================================
# XGBoost 提供了三种方式计算 feature_importances_，分别是：
# 1. Weight（默认模式）：统计某个特征在所有树中被用作分裂节点的次数。
#
# 2. Gain：统计特征在分裂节点上带来的 平均不纯度减少量。  
#
# 3. Cover：统计使用某特征分裂的节点的 平均覆盖率。
# ============================================================

# --- code cell ---
import xgboost as xgb

train= xgb.DMatrix(data= Xtrain, label= Ytrain)
XGB= xgb.train(params= {}, dtrain= train)

# --- code cell ---
XGB.get_score(importance_type= 'weight')

# ============================================================
# ### LightGBM
# ============================================================

# ============================================================
# LightGBM 提供了两种方式计算：
#
# 1. Split（默认模式）：统计特征在所有树中被用作分裂节点的次数。（类似于 XGBoost 的 Weight ）
# 2. Gain：统计特征在所有树中带来的 不纯度减少总量。（类似于 XGBoost 的 Gain，但不取平均值）
# ============================================================

# --- code cell ---
import lightgbm

train= lightgbm.Dataset(Xtrain, label= Ytrain)
lgb = lightgbm.train(params={}, train_set= train)

# --- code cell ---
lgb.feature_importance(importance_type= 'split')

# ============================================================
# ## SHAP（后处理的方法）
# ============================================================

# ============================================================
# SHAP值的主要思想是Shapley值，Shapley值是一个来自合作博弈论（coalitional game theory）的方法，是一种根据玩家对总支出的贡献来为玩家分配支出的方法。用shaply值去解释机器学习的预测的话，其中“总支出”就是数据集单个实例的模型预测值，“玩家”是特征，“收益”是该实例的实际预测减去所有实例的平均预测。 
#
# Shapley 值的一个基本特性是，它们的总和始终等于所有玩家在场时的游戏结果与没有玩家在场时的游戏结果之间的差异。对于机器学习模型，这意味着所有输入特征的 SHAP 值将始终等于 基线模型输出 与 当前模型输出 之间的差异。
# ============================================================

# --- code cell ---
# 在SHAP中进行模型解释需要先创建一个explainer，SHAP支持很多类型的explainer(例如deep, gradient, kernel, linear, tree, sampling)
import shap
explainer = shap.Explainer(XGB)
shap_values = explainer(Xtrain)

# --- code cell ---
shap_values

# --- code cell ---
# 查看特征重要性
shap.plots.bar(shap_values, max_display=10)

# --- code cell ---
# 查看某个样本预测值的特征贡献
shap.plots.bar(shap_values[1], max_display=10)

# --- code cell ---
# 查看全体样本的特征贡献（蜂群图）
shap.plots.beeswarm(shap_values)

# ============================================================
# # 特征影响量化分析
# ============================================================

# ============================================================
# 有些时候，我们需要解释某个特征(X)针对Y的具体影响，即 当X每变动一个单位，Y变动多少个单位
# ============================================================

# --- code cell ---
# 单个特征的偏依赖图
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(
    estimator=rf,
    X=Xtrain,
    features=['GEND','BANK','age','meda'],
    subsample=1000    # 样本采样数量
)

# --- code cell ---
# 偏依赖的数值计算（计算复杂度高，建议每次只计算一个特征）
from sklearn.inspection import partial_dependence
partial_dependence(
    estimator=rf,
    X=Xtrain,
    features=['age'],  # 'GEND','BANK','meda',
)

# ============================================================
# # 决策路径分析
# ============================================================

# ============================================================
# 在某些领域，我国对于模型透明度有着相应的规章制度。此时，决策路径分析在这些场景中非常有用，特别是在需要解释机器学习模型的决策过程时。例如：
# 1. 银行在评估贷款申请时使用了机器学习模型来判断是否可以给申请人放贷，决策路径分析可以精确展示为什么该用户通过/未通过审批。
# 2. 医院使用基于机器学习模型来预测患者是否患有某种疾病，决策路径分析可以精确展示模型如何根据患者的血糖水平、体重指数、年龄等特征做出诊断。
# ============================================================

# ============================================================
# ## 决策树可视化
# ============================================================

# --- code cell ---
# 简单的决策树可以直接使用sklearn+matplotlib可视化（示例展示）
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

# --- code cell ---
# 对于更加复杂的决策树，可视化树很难判断具体决策路径，因此我们引入一个新的方法

# ============================================================
# ## 决策路径
# ============================================================

# ============================================================
# ### 单棵决策树
# ============================================================

# --- code cell ---
# 需要分析决策路径的预测样本
sample= np.array(Xtest.loc[1])

# 获取决策路径
node_indicator= dt.decision_path(sample.reshape(1, -1))
node_index= node_indicator.indices

# 获取树的分裂特征和阈值（-2表示叶节点）
feature= dt.tree_.feature
threshold= dt.tree_.threshold

# --- code cell ---
# 打印样本的详细决策路径
print("决策路径:")
for node_id in node_index:
    if feature[node_id] != -2:  # 非叶节点
        feature_name = dt.feature_names_in_[feature[node_id]]
        threshold_value = threshold[node_id]
        sample_value = sample[feature[node_id]]
        print(f"节点 {node_id}: 如果 {feature_name} <= {threshold_value:.2f} (样本值: {sample_value:.2f})")
    else:
        print(f"叶节点 {node_id}: 预测结果 = {dt.classes_[dt.tree_.value[node_id].argmax()]}")

# ============================================================
# ### 随机森林
# ============================================================

# --- code cell ---
# 训练一个只有5棵树的森林，方便演示
rf_clf= RandomForestClassifier(n_estimators=5, n_jobs= -1).fit(Xtrain, Ytrain)

for i, tree in enumerate(rf_clf.estimators_):
    print(f"第 {i+1} 棵树的决策路径:")
    # 获取决策路径
    node_indicator = tree.decision_path(sample.reshape(1, -1))
    node_index = node_indicator.indices

    # 获取树的分裂特征和阈值
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    for node_id in node_index:
        if feature[node_id] != -2:  # 非叶节点
            feature_name = rf_clf.feature_names_in_[feature[node_id]]
            threshold_value = threshold[node_id]
            sample_value = sample[feature[node_id]]
            print(f"  节点 {node_id}: 如果 {feature_name} <= {threshold_value:.2f} (样本值: {sample_value:.2f})")
        else:
            leaf_value = tree.tree_.value[node_id]
            print(f"  叶节点 {node_id}: 分类结果 = {np.argmax(leaf_value)}, 节点值 = {leaf_value}\n")

# ============================================================
# XGBoost和LightGBM暂不支持决策路径的输出
# ============================================================

# ============================================================
# # 规则提取
# ============================================================

# ============================================================
# 某些场景下，我们需要将机器学习模型转化为业务逻辑（如硬编码）。例如，当我们需要将模型迁移到某个不支持机器学习模型的生产系统中时。
#
# 需要做规则提取的场景，一般都需要模型足够简单。
# ============================================================

# --- code cell ---
# 取重要性前5的特征，重新训练一棵简单的决策树
feature_importance= pd.DataFrame({'Feature':dt.feature_names_in_, 'Importance':dt.feature_importances_})
feature_sort= feature_importance.sort_values(by='Importance', ascending=False)

Xtrain_rule= Xtrain[feature_sort.head(5)['Feature']]

dt_rule= DecisionTreeClassifier(max_depth=5).fit(Xtrain_rule, Ytrain)

# --- code cell ---
# 提取规则
from sklearn.tree import export_text

rules = export_text(dt_rule, feature_names= dt_rule.feature_names_in_.tolist())
print(rules)

# --- code cell ---
# 若决策树足够简单，也可以通过可视化图来提取规则
plt.figure(figsize=(12, 8))
plot_tree(dt_rule, feature_names=dt_rule.feature_names_in_.tolist(), class_names=['0','1'], filled=True)
plt.show()

# ============================================================
# # 反事实解释（Counterfactual Explanation）
# ============================================================

# ============================================================
# 在某些场景下，我们需要回答：对于某个样本，如何通过改变某些输入特征的值，更改模型预测结果。例如：
# 1. 金融行业：“如果你的收入提高到 X1 或信用分数增加到 X2，你的贷款申请将被批准。”
# 2. 医疗行业：“如果某项指标（如 BMI）降低到 X1 或血糖控制在 X2，你的患病评估结果会下降。”
# 3. 人力资源行业：“如果你拥有 X1 年的相关工作经验或掌握技能 X2，你可能会被选中。”
# ============================================================

# --- code cell ---
# 注意：安装此库会将pandas的版本自动降为1.5，需要手动回退，请谨慎安装
# !pip install dice_ml

# --- code cell ---
import dice_ml
train_dataset= pd.concat([Xtrain, Ytrain], axis=1)

# 生成dice专用格式的数据集
d = dice_ml.Data(dataframe=train_dataset, continuous_features=num_cols.columns.tolist(), outcome_name='resp_flag')

# 将模型导入dice
m = dice_ml.Model(model=dt, backend="sklearn")

# 创建解释器
exp = dice_ml.Dice(d, m, method="random")

# --- code cell ---
# 生成反事实样本
e1 = exp.generate_counterfactuals(Xtest[0:1], total_CFs=4, desired_class="opposite", random_seed=42)
e1.visualize_as_dataframe(show_only_changes=True)

# --- code cell ---

