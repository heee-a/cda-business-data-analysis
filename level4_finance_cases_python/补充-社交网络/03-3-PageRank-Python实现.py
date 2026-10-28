"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
#PageRank-Python实现

# --- code cell ---
import numpy as np
import random

# --- code cell ---
#############################################
#############################################
#########  第一部分：简单测试
#############################################
#############################################

# --- code cell ---
#以不本篇简单例子为例
#初始矩阵M

# --- code cell ---
M =np.array([[0,1/2,0,1/2],
   [1/3,0,0,1/3],
   [1/3,1/2,0,0],
   [1/3,0,1,0]])

# --- code cell ---
M

# --- code cell ---
v =np.array( [[1/4],[1/4],[1/4],[1/4]])

# --- code cell ---
v

# --- code cell ---
Mv1 = np.dot(M,v)

# --- code cell ---
Mv1

# --- code cell ---
def MV_computer(N):
    for i in range(N):
        Mv = np.dot(M,v)
        i = i+1   
    return Mv    

# --- code cell ---
Result = MV_computer(1000)

# --- code cell ---
Result

# --- code cell ---
#############################################
#############################################
#########  第二部分：随机测试
#############################################
#############################################

# --- code cell ---
def PR_Gdata(N, alpha=0.8): # random > alpha, then here is a edge.
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if random.random() < alpha:
                G[i][j] = 1
    return G
G1 = PR_Gdata(10)

# --- code cell ---
G1

# --- code cell ---
def pagerank_GtoM(G1, N):
    M = np.zeros((N, N))
    for i in range(N):
        D_i = sum(G1[i])
        if D_i == 0:
            continue
        for j in range(N):
            M[j][i] = G1[i][j] / D_i # 边和链接矩阵转换
    return M
M1 = pagerank_GtoM(G1, 10)


# --- code cell ---
def PageRank(M, N, T=300, eps=1e-6):
    R = np.ones(N) / N
    for time in range(T):
        R_new = np.dot(M, R)
        if np.linalg.norm(R_new - R) < eps:
            break
        R = R_new.copy()
    return R_new


# --- code cell ---
Result = PageRank(M=M1,N=10)

# --- code cell ---
Result = PageRank(M,4,T=300, eps=1e-6)

# --- code cell ---
Result

# --- code cell ---

