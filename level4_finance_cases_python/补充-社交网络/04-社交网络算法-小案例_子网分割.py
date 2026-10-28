"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
import networkx as nx  
import pandas as pd
import matplotlib.pyplot as plt

edge_list=pd.read_csv(r".\data\stormofswords.csv")

edge_list.head()

edge_list.Weight=edge_list.Weight.astype(int)


def plot_network(pd_table):
    '''
    绘制网络关系图
    '''
    G = nx.from_pandas_edgelist(edge_list, source = 'Source', target = 'Target', 
                                edge_attr = 'Weight',create_using = nx.DiGraph())
    nx.draw(G,
        #pos = nx.spiral_layout(G),
        node_color = 'b', 
        edge_color = 'r',
        with_labels = True,
        font_size =10,
        node_size =200,)
    plt.show()

# --- code cell ---
plot_network(edge_list)

# --- code cell ---
G=nx.from_pandas_edgelist(edge_list,'Source',"Target",edge_attr=True)

# --- code cell ---
import networkx as nx
import community
#from community import  community_louvain

# --- code cell ---
partition = community.best_partition(G)

# --- code cell ---
partition

# --- code cell ---
len(partition)

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

