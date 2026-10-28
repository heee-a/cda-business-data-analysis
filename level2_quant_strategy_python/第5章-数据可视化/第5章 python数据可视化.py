"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# 2.1.1 线图
# ============================================================

# --- code cell ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(r"2017股价.xlsx")
df

# --- code cell ---
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['axes.unicode_minus']=False#解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei']#指定默认字体 

# --- code cell ---
plt.figure(figsize=(10,4)) #生成画布并设置画布尺寸
plt.plot(range(15),df.loc[:,"收盘价"],'-*') #共15个数据点，x轴需要15个刻度点
plt.plot(range(15),df.loc[:,"开盘价"],'-o') 
plt.plot(range(15),df.loc[:,"最高价"],'-v') 
plt.plot(range(15),df.loc[:,"最低价"],'-^')
plt.title('股价走势图') #设置标题
plt.xlabel('时间') #设置x轴标签
plt.ylabel('股价') #设置y轴标签
plt.xticks(range(15),df.日期,rotation=30) #设置刻度的标签，倾斜度
plt.legend(['收盘价','开盘价','最高价','最低价']) #设置图例
plt.show()#展示图形

# ============================================================
# 2.1.2 饼图
# ============================================================

# --- code cell ---
np.random.seed(1)
df=pd.DataFrame(data={"方案":["plan A","plan B","plan C","plan D","plan E"]
                            ,"投票人数":np.random.randint(10,100,5)})
df

# --- code cell ---


plt.figure(figsize=(6,6))
plt.pie(df.投票人数 #饼图数据源
        ,labels=df.方案 #设置每个扇形对应的标签
        ,autopct='%.2f%%' #设置百分比经度
        ,textprops={'fontsize':13} #设置饼图中字体的大小
    )
plt.title('方案投票占比饼图',fontsize=14);
plt.show()

# ============================================================
# 2.1.3 条形图
# ============================================================

# --- code cell ---
plt.figure(figsize=(8,4))
plt.bar(df.方案,df.投票人数,width=0.6)
plt.title("各方案投票人数柱形图",fontsize = 14)
plt.ylabel("投票人数",fontsize = 13)
plt.xlabel("方案",fontsize = 13)
plt.show()

# --- code cell ---


plt.figure(figsize=(8,4))
plt.barh(df.sort_values("投票人数").方案,df.sort_values("投票人数").投票人数,height=0.6)
plt.xlabel("投票人数",fontsize = 13)
plt.ylabel("方案",fontsize = 13)
plt.title("各方案投票人数条形图",fontsize = 14)
plt.show()

# ============================================================
# 2.1.4 直方图
# ============================================================

# --- code cell ---
np.random.seed(1)
np.random.seed(1)
data = np.random.normal(loc=80, scale=20,size=200).astype(int)
data

# --- code cell ---
plt.hist(data,[30,40,50,60,70,80,90,100,110,120,130])
plt.xticks([30,40,50,60,70,80,90,100,110,120,130])
plt.show()

# ============================================================
# 2.1.5 散点图
# ============================================================

# --- code cell ---
df = pd.DataFrame({"产量":[2200,3300,4400,5500,5700,7300,8000]
              ,"降水量":[22,39,55,66,105,100,122]})
df

# --- code cell ---
plt.scatter(df.降水量,df.产量)
plt.show()

# ============================================================
# 2.1.6 箱线图
# ============================================================

# --- code cell ---
np.random.seed(1)
df=pd.DataFrame({"语文":np.random.randint(60,150,10)
                ,"英语":np.random.randint(40,150,10)
                ,"数学":np.random.randint(20,150,10)})
df


# --- code cell ---
plt.figure(figsize=(8,6))
plt.boxplot([df.语文,df.英语,df.数学]
            ,labels=['语文','英语','数学']);

# ============================================================
# 2.2.1 带核密度估计的直方图
# ============================================================

# --- code cell ---
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(1)
np.random.seed(1)
data = np.random.normal(loc=80, scale=20,size=200).astype(int)
data

# --- code cell ---
sns.displot(data,bins=[30,40,50,60,70,80,90,100,110,120,130],kde=True)# kde=True 显示核密度估计曲线
plt.show()

# ============================================================
# 2.2.2 二元分布图
# ============================================================

# --- code cell ---
np.random.seed(0)
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
df.head()

# --- code cell ---
sns.jointplot(x="x", y="y", data=df)
plt.show()

# --- code cell ---
sns.relplot(x="x", y="y", data=df)
plt.show()

# --- code cell ---
sns.jointplot(x="x", y="y", data=df, kind="kde")
plt.show()

# --- code cell ---
sns.displot(data=df, x="x", y="y", kind="kde")
plt.show()

# ============================================================
# 2.2.3 热图
# ============================================================

# --- code cell ---
np.random.seed(0)
df0=pd.DataFrame(np.random.uniform(
                -1,1,(7,7)),index=list("abcdefg"),columns=list("ABCDEFG"))
df0

# --- code cell ---
plt.figure(figsize=(8,6))
sns.set(font_scale=1.5)#设置字体大小
sns.heatmap(df0)

# --- code cell ---


# --- code cell ---


# --- code cell ---


# --- code cell ---


# --- code cell ---


# --- code cell ---


# --- code cell ---


# --- code cell ---


# --- code cell ---

