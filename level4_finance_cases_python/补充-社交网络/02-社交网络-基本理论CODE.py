"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
#1.导入相应的包
import networkx as nx
import matplotlib.pyplot as plt

# --- code cell ---
##======================================================
##                        无向图                       ##
##======================================================
##1.1添加无向图
##--定义无向图 
G = nx.Graph() 

# --- code cell ---
##--定义图的节点
nodes=['0','1','2','3','4','5','6','7','8'] 

# --- code cell ---
##--增加图添加节点
G.add_nodes_from(nodes) 

nx.draw_networkx(G,node_color = 'g',with_labels = True,font_color='w',font_size =8,node_size =200)

plt.show()

# --- code cell ---
##--增加图边
edges=[('0','0',1),('0','1',1),('0','5',1),('0','5',2),('1','2',3),('1','4',5),('2','1',7),('2','4',6),('3','5',5),('4','8',4),('6','7',1),('7','8',2),('8','6',3),('8','3',5)] 

# --- code cell ---
##--展示无向图
G.add_weighted_edges_from(edges) 
nx.draw_networkx(G,node_color = 'g',with_labels = True,font_color='w',font_size =8,node_size =200) #详细见draw_networkx介绍
plt.show()

# --- code cell ---
##======================================================
##                        有向图                       ##
##======================================================

# --- code cell ---
#清除无向图 
G.clear() 
 
#定义有向图 
G = nx.DiGraph() 
#添加节点和边 
G.add_nodes_from(nodes) 
G.add_weighted_edges_from(edges) 
nx.draw_networkx(G,node_color = 'g',with_labels = True,font_color='w',font_size =8,node_size =200) #详细见draw_networkx介绍
plt.show()

# --- code cell ---
##======================================================
##                        计算度                      ##
##======================================================

# --- code cell ---
G.out_degree('1')

# --- code cell ---


# --- code cell ---


# --- code cell ---

