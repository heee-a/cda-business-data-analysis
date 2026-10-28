"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
import pandas as pd
accepts =pd.read_csv ( 'accepts.csv' )
cross_table=pd.crosstab(accepts.bankruptcy_ind,accepts.bad_ind, margins=True)
cross_table

# --- code cell ---
cross_table_rowpct = cross_table.div(cross_table ["All"] ,axis = 0)
cross_table_rowpct

# --- code cell ---
# 打印结果
from scipy import stats
print(
    "chisq = %6.4f\np-value = %6.4f\ndof = %d\nexpected_freq = %s"
    %stats.chi2_contingency (cross_table))


# --- code cell ---

