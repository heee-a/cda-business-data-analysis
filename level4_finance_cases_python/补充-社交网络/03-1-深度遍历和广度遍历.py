"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
from collections import deque

# --- code cell ---
class Graph:
    """无向图"""
    def __init__(self, num_vertices):
        # 顶点数量
        self._num_vertices = num_vertices
        # [[], [], [] ...]，内部每个小列表存储该顶点的相连的其他顶点
        self._adjacency = [[] for _ in range(num_vertices)]
    def add_edge(self, s, t):
        """无向图一条边存两次"""
        self._adjacency[s].append(t)
        self._adjacency[t].append(s)
    def _generate_path(self, s, t, prev):
        # 用or会打印第一个顶点，用and则不会
        if prev[t] or s != t:
            yield from self._generate_path(s, prev[t], prev)
        yield str(t)
        
    def bfs(self, s, t):
        """通过广度优先搜索，打印s -> t的路径"""
        if s == t:
            return
        # 定义文中提到的的三个重要变量，并将第一个顶点s加入各变量中
        visited = [False] * self._num_vertices
        visited[s] = True
        q = deque()
        q.append(s)
        prev = [None] * self._num_vertices
        
        while q:
            v = q.popleft()
            # 遍历顶点v相连的其他顶点，并将每个顶点加入到队列中继续遍历
            for neighbour in self._adjacency[v]:
                # 各顶点未被遍历
                if not visited[neighbour]:
                    # 记录path
                    prev[neighbour] = v
                    # 符合终止条件则打印
                    if neighbour == t:
                        print(' -> '.join(self._generate_path(s, t, prev)))
                        return
                    # 记录遍历过的顶点
                    visited[neighbour] = True
                    # 将加入队列，后面会遍历该顶点
                    q.append(neighbour)
        
    def dfs(self, s, t):
        found = False
        visited = [False] * self._num_vertices
        prev = [None] * self._num_vertices
        # 递归
        def _dfs(from_vertex):
            nonlocal found
            if found:
                return
            visited[from_vertex] = True
            if from_vertex == t:
                found = True
                return
            for neighbour in self._adjacency[from_vertex]:
                if not visited[neighbour]:
                    prev[neighbour] = from_vertex
                    _dfs(neighbour)
        _dfs(s)
        print(' -> '.join(self._generate_path(s, t, prev)))

# --- code cell ---
if __name__ == "__main__":
    """
      0 ----- 1 ----- 2
      |     / |     / |
      |    /  |    /  |
      |   /   |   /   |
      |  /    |  /    |
      | /     | /     |
      3 ----- 4 ----- 5
              |     / |
              |    /  |
              |   /   |
              |  /    |
              | /     |
              6 ----- 7
    """
    graph = Graph(8)
    graph.add_edge(0, 1)
    graph.add_edge(0, 3)
    graph.add_edge(1, 2)
    graph.add_edge(1, 4)
    graph.add_edge(2, 5)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(4, 6)
    graph.add_edge(5, 7)
    graph.add_edge(6, 7)

    print(graph._adjacency)
    # 输出结果：[[1, 3], [0, 2, 4], [1, 5], [0, 4], [1, 3, 5, 6], [2, 4, 7], [4, 7], [5, 6]]
    graph.bfs(0, 7)
    # 输出结果：0 -> 1 -> 2 -> 5 -> 7
    graph.dfs(0, 7)
    # 输出结果：0 -> 1 -> 2 -> 5 -> 4 -> 6 -> 7 
