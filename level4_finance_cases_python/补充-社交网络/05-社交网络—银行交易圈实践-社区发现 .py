"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# #### 社区发现
# ============================================================

# --- code cell ---
#pip install community
#pip install python-louvain
import pandas as pd
import numpy as np
import os
import networkx as nx
import community
#from community import  community_louvain

# --- code cell ---
Original_data = pd.read_csv('data/Archive-社交网络-交易对手明细汇总.csv',index_col=0,header=0)

# --- code cell ---
Original_data.head(10)

# --- code cell ---
Original_data.drop(["统计日期","近六个月转入次数","近六个月转入金额","近六个月转出次数","近六个月转出金额","近六个月交易次数","近六个月交易金额","近六个月交易月份数","近六个月月均交易次数","近六个月月均交易金额","近六个月笔均交易金额","近六个月单笔最大交易金额","近六个月单笔最小交易金额"],inplace=True,axis=1)

# --- code cell ---
Original_data.head(10)

# --- code cell ---
G=nx.from_pandas_edgelist(Original_data,"源客户号","目标客户号",edge_attr=True)

# --- code cell ---
G

# --- code cell ---
partition = community.best_partition(G)

# --- code cell ---
partition

# --- code cell ---
len(set(partition.values()))

# --- code cell ---
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

# --- code cell ---
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()

# --- code cell ---

