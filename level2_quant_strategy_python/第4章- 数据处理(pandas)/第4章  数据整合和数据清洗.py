"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# 1.1.1读取数据
# ============================================================

# --- code cell ---
import pandas as pd
csv = pd.read_csv('sample.csv')
csv

# --- code cell ---
csv = pd.read_csv('sample.csv',\
                   usecols=['id','name'],\
                   nrows=2) #读取'id'和'name'两列，仅读取前两行
csv


# --- code cell ---
csvs = pd.read_csv('data/big.csv',chunksize=900)
for i in csvs:
    print (i.shape)


# --- code cell ---
csvs = pd.read_csv('data/big.csv',chunksize=900)
dat = pd.concat(csvs,ignore_index=True)
dat.shape


# --- code cell ---
csv = pd.read_csv('sample.csv',
                  na_values='99999')
csv


# --- code cell ---
csv = pd.read_csv('sample.csv',
                  encoding='utf-8')
csv


# --- code cell ---
csv.to_csv('write.csv',encoding='utf-8',index=False)

# ============================================================
# 1.2.1 行列操作
# ============================================================

# --- code cell ---
import numpy as np
import pandas as pd
np.random.seed(0) #为保证代码每次运行都能得到相同的一组数据，设置随机数种子
sample = pd.DataFrame(np.random.randn(4,5)
                     ,columns=list("abcde")
                     )
sample

# --- code cell ---
sample['a']

# --- code cell ---
sample.iloc[:,0]

# --- code cell ---
sample.loc[:,"a"]

# --- code cell ---
sample[['a']]

# --- code cell ---
sample.loc[:1,:"b"]

# --- code cell ---
sample.iloc[0:2,0:2]

# --- code cell ---
sample['new_col'] = sample['a'] - sample['b']
sample

# --- code cell ---
sample.assign(new_col2 = sample['a'] - sample['b'],
                  new_col3 = sample['a'] + sample['b'])


# --- code cell ---
sample.drop(columns='a')

# --- code cell ---
sample.drop(columns=['a','b'])

# ============================================================
# 1.2.2 条件查询
# ============================================================

# --- code cell ---
sample =pd.DataFrame({'group':[1,1,1,2,1,2]
                      ,'name':['Bob','Lindy','Mark','Miki','Sully','Rose']
                      ,'score':[98,78,87,77,65,67]
                      ,})
sample

# --- code cell ---
sample.score > 70

# --- code cell ---
sample[sample.score > 70]

# --- code cell ---
sample[(sample.score > 70) & (sample.group ==1)] #且

# --- code cell ---
sample[~(sample.group ==1)] #非

# --- code cell ---
sample[(sample.group ==2) | (sample.group ==1)]

# --- code cell ---
sample.query('score > 90')

# --- code cell ---
sample.query('(group ==2) |(group == 1)')

# --- code cell ---
# sample[sample['score'].between(70,80),inclusive=True]
sample[sample['score'].between(70, 80, inclusive='both')]#Pandas 1.3.0 版本开始，inclusive 参数被修改为 inclusive='both'（或’neither’/‘left’/‘right’） 的字符串形式

# --- code cell ---
sample[sample['name'].isin(['Bob','Lindy'])]

# --- code cell ---
sample[sample['name'].str.contains('[M]+')]

# ============================================================
# 1.2.3 横向连接
# ============================================================

# --- code cell ---
df1 = pd.DataFrame({ 'col1':['a','b','c'],
                     'id':[1,2,3],
                   })
df2 = pd.DataFrame({ 'col2':['d','e'],
                    'id':[4,3],
                   })

# --- code cell ---
df1

# --- code cell ---
df2

# --- code cell ---
df1.merge(df2,how='inner',on='id')

# --- code cell ---
df1.merge(df2,how='inner',left_on='id',right_on='id')

# --- code cell ---
df1.merge(df2,how='left',on='id')

# --- code cell ---
df1.merge(df2,how='right',on='id')

# --- code cell ---
df1.merge(df2,how='outer',on='id')

# --- code cell ---
df1 = pd.DataFrame({'id1':[1,2,3],
                    'col1':['a','b','c']},
                  index = [1,2,3])
df2 = pd.DataFrame({'id2':[1,2,3],
                    'col2':['aa','bb','cc']},
                  index = [1,3,2])

# --- code cell ---
df1

# --- code cell ---
df2

# --- code cell ---
pd.concat([df1,df2],axis=1)

# --- code cell ---
df1.join(df2)

# ============================================================
# 1.2.4 纵向合并
# ============================================================

# --- code cell ---
df1 = pd.DataFrame({'id':[1,1,1,2,3,4,6],
                    'col':['a','a','b','c','v','e','q']})
df2 = pd.DataFrame({'id':[1,2,3,3,5],
                    'col':['x','y','z','v','w']})

# --- code cell ---
df1

# --- code cell ---
df2

# --- code cell ---
pd.concat([df1,df2],ignore_index=True,axis=0)

# --- code cell ---
pd.concat([df1,df2],ignore_index=True).drop_duplicates()

# --- code cell ---
df3 = df1.rename(columns = {'col':'new_col'})
df3

# --- code cell ---
pd.concat([df1,df3],ignore_index=True).drop_duplicates()

# ============================================================
# 1.2.5 排序
# ============================================================

# --- code cell ---
sample=pd.DataFrame({'group':[1,1,1,2,1,2],
                    'name':['Bob','Lindy','Mark','Miki','Sully','Rose'],
                    'score':[98,78,87,77,77,np.nan],
                    })
sample

# --- code cell ---
sample.sort_values('score',ascending=False,na_position='last')

# --- code cell ---
sample.sort_values(['group','score']) 

# ============================================================
# 1.2.6 分组汇总
# ============================================================

# --- code cell ---
sample = pd.DataFrame({"name":['Bob','Lindy','Mark','Miki','Sully','Rose']
                        ,"class":[1,1,1,2,1,2]
                        ,"grade":[1,1,1,2,2,2]
                        ,"chinese":[88,78,86,56,77,54]
                        ,"math":[98,78,87,77,77,np.nan]
                        })
sample

# --- code cell ---
sample.groupby('grade')[['math']].max()

# --- code cell ---
sample.groupby(['grade','class'])[['math']].mean()

# --- code cell ---
sample.groupby(['grade']) [['math','chinese']].mean()

# --- code cell ---
sample.groupby('class')['math'].agg(['mean','min','max'])

# --- code cell ---
df = sample.groupby(['grade','class']) [['math','chinese']].agg(['min','max'])
df

# --- code cell ---
df['math']['min']

# --- code cell ---
df.loc[:,('math','min')]

# --- code cell ---
df.loc[(2,2),('math','min')]

# ============================================================
# 1.2.7 拆分列
# ============================================================

# --- code cell ---
table = pd.DataFrame({ 'Monetary':[3608,420,1894,3503,4567],
                        'cust_id':[10001,10001,10002,10002,10003],
                        'type':['Normal','Special_offer',\
                        'Normal','Special_offer','Special_offer']
                     })
table 

# --- code cell ---
pd.pivot_table(table,index='cust_id',columns='type',values='Monetary')

# --- code cell ---
pd.pivot_table(table,index='cust_id',columns='type',values='Monetary',
        fill_value=0,aggfunc='sum')

# ============================================================
# 1.2.8 赋值与条件赋值
# ============================================================

# --- code cell ---
sample = pd.DataFrame({'name':['Bob','Lindy','Mark',
        'Miki','Sully','Rose'],
        'score':[99,78,999,77,77,np.nan],
        'group':[1,1,1,2,1,2],})
sample

# --- code cell ---
sample.score.replace(999,np.nan)

# --- code cell ---
sample.replace({'score':{999:np.nan},
                'name':{'Bob':np.nan}})


# --- code cell ---
sample

# --- code cell ---
def transform(row):
    if row['group'] == 1:
        return ('class1')
    elif row['group'] == 2:
        return ('class2')  

sample.apply(transform,axis=1)

# --- code cell ---
sample.assign(class_n = sample.apply(transform,axis=1))

# --- code cell ---
sample = sample.copy()
sample.loc[sample.group==1,'class_n']='class1'
sample.loc[sample.group==2,'class_n']='class2'
sample

# ============================================================
# 1.3.1 重复值处理
# ============================================================

# --- code cell ---
sample = pd.DataFrame({'group':[1,1,1,2,1,2],
                        'id':[1,1,1,3,4,5],
                       'name':['Bob','Bob','Mark','Miki','Sully','Rose'],
                       'score':[99,99,87,77,77,np.nan],
                       })
sample

# --- code cell ---
sample[sample.duplicated()]

# --- code cell ---
sample.drop_duplicates()

# --- code cell ---
sample.drop_duplicates('id')

# ============================================================
# 1.3.2 缺失值处理
# ============================================================

# --- code cell ---
sample = pd.DataFrame({'group':[1,1,np.nan,2,1,np.nan],
                        'id':[1,1,1,3,4,np.nan],
                       'name':['Bob','Bob','Mark','Miki','Sully',np.nan],
                       'score':[99,np.nan,87,77,77,np.nan],
                       })
sample

# --- code cell ---
sample.apply(lambda col:sum(col.isnull())/col.size)

# --- code cell ---
sample.score.fillna(sample.score.mean())

# --- code cell ---
sample.score.fillna(sample.score.median())

# --- code cell ---
sample.score.isnull()

# --- code cell ---
sample.score.isnull().apply(int)

# ============================================================
# 1.3.3 噪声值处理
# ============================================================

# --- code cell ---
def cap(x,quantile=[0.01,0.99]):
    """盖帽法处理异常值
    Args：
        x：pd.Series列，连续变量
        quantile：指定盖帽法的上下分位数范围
    """

# 生成分位数
    Q01,Q99=x.quantile(quantile).values.tolist()

# 替换异常值为指定的分位数
    if Q01 > x.min():
        x = x.copy()
        x.loc[x<Q01] = Q01

    if Q99 < x.max():
        x = x.copy()
        x.loc[x>Q99] = Q99

    return(x)


# --- code cell ---
sample = pd.DataFrame({'normal':np.random.randn(1000)})
sample.hist(bins=50)

# --- code cell ---
new = sample.apply(cap,quantile=[0.01,0.99])
new.hist(bins=50)

# --- code cell ---
np.random.seed(0)
sample =pd.DataFrame({'normal':np.random.randn(10)})
sample

# --- code cell ---
pd.cut(sample.normal,5)

# --- code cell ---
pd.cut(sample.normal,bins=5,labels=[1,2,3,4,5])

# --- code cell ---
pd.cut(sample.normal,bins=2,labels=['bad','good'])

# --- code cell ---
sample.normal.quantile([0,0.5,1])

# --- code cell ---
pd.cut(sample.normal,bins=sample.normal.quantile([0,0.5,1]),
 include_lowest=True)

# --- code cell ---
pd.cut(sample.normal,bins=sample.normal.quantile([0,0.5,1]),labels=['bad','good'],include_lowest=True)

# --- code cell ---


# --- code cell ---


# --- code cell ---

