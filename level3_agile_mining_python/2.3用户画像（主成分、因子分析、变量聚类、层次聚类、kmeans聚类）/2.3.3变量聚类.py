"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# ### cities_10记录了十个沿海省份的经济指标，通过降维方法对各省、市经济发展情况
#
# 变量含义如下：
#
# |变量|含义|
# |:--:|:--:|
# |AREA|地区|
# |X1|GDP|
# |X2|人均GDP|
# |X3|工业增加值|
# |X4|第三产业增加值|
# |X5|固定资产投资|
# |X6|基本建设投资|
# |X7|社会消费品零售总额|
# |X8|海关出口总额|
# |X9|地方财政收入|
# ============================================================

# --- code cell ---
import sys
sys.path.append("./myscript")

import pandas as pd
from varSelec import var_select

# 读取数据
model_data = pd.read_csv("cities_10.csv", encoding='gbk')
model_data.head()
data = model_data.loc[ :,'X1':]

# 变量聚类，并选择保留k个变量
selected_data = var_select(data, k=2)
selected_data

# --- code cell ---

