"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 单因素方差分析
# ============================================================

# --- code cell ---
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd

# --- code cell ---
data = pd.read_csv('creditcard_exp.csv')
df = data[['avg_exp','edu_class','gender']].dropna(how='any',axis=0)
df['edu_class'] = df['edu_class'].astype('str')
df['gender'] = df['gender'].astype('str')

# --- code cell ---
df['avg_exp'] = df['avg_exp'].apply(lambda x: 0 if x==' ' else x)
df['avg_exp'] = df['avg_exp'].astype('float')

# --- code cell ---
formula = f'avg_exp ~ C(edu_class) + C(gender)'
anova = ols(formula=formula, data=df).fit()
anova_results = anova.summary()
print(anova_results)

# ============================================================
# # 多因素方差分析
# ============================================================

# --- code cell ---
data = pd.read_csv('creditcard_exp.csv')
df = data[['avg_exp','edu_class','gender']].dropna(how='any',axis=0)
df['edu_class'] = df['edu_class'].astype('str')
df['gender'] = df['gender'].astype('str')
df['avg_exp'] = df['avg_exp'].apply(lambda x: 0 if x==' ' else x)
df['avg_exp'] = df['avg_exp'].astype('float')

# --- code cell ---
formula = f'avg_exp ~ C(edu_class) + C(gender) + C(edu_class)*C(gender)'

anova = ols(formula=formula, data=df).fit()
anova_results = anova.summary()
print(anova_results)

# --- code cell ---


# --- code cell ---

