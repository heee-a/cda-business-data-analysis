"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
##======================================================
##                        有向图                       ##
##======================================================

# --- code cell ---
#1.导入相应的包
import networkx as nx
import matplotlib.pyplot as plt

# --- code cell ---
##1.1添加无向图
##--定义无向图 
G = nx.Graph() 

# --- code cell ---
##--定义图的节点
nodes=['1','2','3','4','5','6','7'] 

# --- code cell ---
##--增加图添加节点
G.add_nodes_from(nodes) 

nx.draw_networkx(G,node_color = 'g',with_labels = True,font_color='w',font_size =8,node_size =200)

plt.show()

# --- code cell ---
##--增加图边
edges=[('1','2',1),('1','6',1),('1','7',1),('2','3',1),('3','4',1),('4','5',1),('4','7',1),('5','6',1),('6','7',1),('7','5',1)] 

# --- code cell ---
#定义有向图 
G = nx.DiGraph() 
#添加节点和边 
G.add_nodes_from(nodes) 
G.add_weighted_edges_from(edges) 
nx.draw_networkx(G,node_color = 'g',with_labels = True,font_color='w',font_size =8,node_size =200) #详细见draw_networkx介绍
plt.show()

# --- code cell ---
#度
nx.degree(G)

# --- code cell ---
#出度
G.out_degree(G)

# --- code cell ---
#入度
G.in_degree(G)

# --- code cell ---
##======================================================
##                        无向图                       ##
##======================================================

# --- code cell ---
#清除无向图 
G.clear() 

# --- code cell ---
##1.1添加无向图
##--定义无向图 
G = nx.Graph() 

# --- code cell ---
##--定义图的节点
nodes=['A','B','C','D','E','F'] 

# --- code cell ---
##--增加图添加节点
G.add_nodes_from(nodes) 

nx.draw_networkx(G,node_color = 'g',with_labels = True,font_color='w',font_size =8,node_size =200)

plt.show()

# --- code cell ---
##--增加图边
edges=[('A','B',1),('A','C',1),('A','D',1),('A','E',1),('A','F',1)] 

# --- code cell ---
##--展示无向图
G.add_weighted_edges_from(edges) 
nx.draw_networkx(G,node_color = 'g',with_labels = True,font_color='w',font_size =8,node_size =200) #详细见draw_networkx介绍
plt.show()

# --- code cell ---
#指标1：度中心度  m/(n-1)
nx.degree_centrality(G)

# --- code cell ---
#指标2：紧密中心度
nx.closeness_centrality(G)

# --- code cell ---
#指标3：介度中心度
nx.betweenness_centrality(G)

# --- code cell ---
#指标4：特征向量中心度
nx.eigenvector_centrality(G)

# --- code cell ---

