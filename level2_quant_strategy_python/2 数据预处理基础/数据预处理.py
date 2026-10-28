"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
import pandas as pd
import os
import numpy as np
# 导入数据
camp = pd.read_csv(r"teleco_camp_orig.csv")
# 脏数据或数据不正确
import matplotlib.pyplot as plt
plt.hist(camp["AvgIncome"],bins=20)#,normed=True)#查看分布情况

# --- code cell ---
camp["AvgIncome"].describe(include="all")

# --- code cell ---
camp["AvgHomeValue"].describe(include="all")

# --- code cell ---
#这里的 0值应该是缺失值，处理 0值。
camp["AvgIncome"]=camp["AvgIncome"].replace({0:np.NaN})

# --- code cell ---
#像这种外部获取的数据要比较小心，经常出现意义不清晰或这错误值。AvgHomeValue也有这种情况
plt.hist(camp["AvgIncome"],bins=20,range=(camp.AvgIncome.min(),camp.AvgIncome.max()))
#由于数据中存在缺失值 ,需要指定绘图的值域

# --- code cell ---
camp["AvgHomeValue"].describe(include="all")

# --- code cell ---


# --- code cell ---

