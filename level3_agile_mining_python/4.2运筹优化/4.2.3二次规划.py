"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# ### 二次规划
# ============================================================

# --- code cell ---
import cplex

op = cplex.Cplex()
op.read("qp1.lp")
op.solve()
# 输出结果
print(op.solution.get_objective_value())
print(op.solution.get_values())


# --- code cell ---


# --- code cell ---

