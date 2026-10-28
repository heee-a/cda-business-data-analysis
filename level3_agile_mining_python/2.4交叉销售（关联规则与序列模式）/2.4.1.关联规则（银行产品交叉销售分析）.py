"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# ### Apriori
# ============================================================

# --- code cell ---
import pandas as pd

data = pd.read_csv('bank_prod_order.csv',encoding='gbk')
data.head()

# --- code cell ---
baskets = data['PROD'].groupby(data['CSR_ID']).apply(list)
baskets.head()

# --- code cell ---
import sys
sys.path.append('./myscript')
from apriori import arules

rules = arules(baskets, min_support=0.02)
rules.head()

# ============================================================
# 可以根据需要进行后续处理，例如按照confidence和lift过滤后，按support排序
# ============================================================

# --- code cell ---
conf = rules[(rules.confidence > 0.1) & (rules.lift > 1)]
conf.sort_values(by='support', ascending=False).head()

# ============================================================
# 如果有必要，我们也可以通过图形看一下规则在支持度、置信度和提升度上的分布情况
# ============================================================

# --- code cell ---
import matplotlib.pyplot as plt
# %matplotlib inline

ax = plt.subplot()
ax.set_xlabel('lift')
conf.plot.scatter(x="lift", y="support",c="confidence",cmap='binary',
                  figsize=(6,4), ax=ax)

plt.show()

# ============================================================
# ### FP_growth
#
# `pip install Orange3-Associate` first!
#
# [参考文档](https://orange3-associate.readthedocs.io/en/latest/scripting.html)
# ============================================================

# --- code cell ---
mapping = data['PROD'].drop_duplicates().to_dict()  # 生成{编号:产品}映射
inverse_mapping = {v:k for k, v in mapping.items()}     # 生成{产品:编号}映射

data['PROD'] = data['PROD'].replace(inverse_mapping)  # 将产品用编号替代
baskets = data['PROD'].groupby(data['CSR_ID']).apply(list)  
baskets.head()

# --- code cell ---
from orangecontrib.associate.fpgrowth import *  

# 计算关联规则
n_examples=len(data)
itemsets = dict(frequent_itemsets(baskets.tolist(), 0.02))
rules = association_rules(itemsets, min_confidence=0.1)
stats = rules_stats(rules, itemsets, n_examples)

result = pd.DataFrame(stats, columns=['atecedent', 'consequent', 'support', 
                                      'confidence', 'coverage', 'strength', 
                                      'lift', 'leverage'])

# 将整型的项转换为原始产品名称
pid_to_product = lambda fsets: tuple(mapping[fset] for fset in fsets)
result['atecedent'] = result['atecedent'].map(pid_to_product)
result['consequent'] = result['consequent'].map(pid_to_product)
result.head()

# ============================================================
# - atecedent (frozenset) – The LHS of the association rule.
# - consequent (frozenset) – The RHS of the association rule.
# - support (int) – Support as an absolute number of instances.
# - confidence (float) – The confidence percent, calculated as: total_support / lhs_rupport.
# - coverage (float) – Calculated as: lhs_support / n_examples
# - strength (float) – Calculated as: rhs_support / lhs_examples
# - lift (float) – Calculated as: n_examples * total_support / lhs_support / rhs_support
# - leverage (float) – Calculated as: (total_support * n_examples - lhs_support * rhs_support) / n_examples**2
# ============================================================

# --- code cell ---

