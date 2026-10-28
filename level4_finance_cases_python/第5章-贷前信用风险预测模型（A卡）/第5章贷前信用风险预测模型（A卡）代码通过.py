"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 第五章贷前信用风险预测模型（A卡）代码
# ============================================================

# ============================================================
# # 5.1智能信贷审批基本框架
# ============================================================

# ============================================================
# # 5.2 特征工程
# ============================================================

# ============================================================
# # 5.3 模型构建与评估
# ============================================================

# ============================================================
# # 5.4 模型监控
# ============================================================

# ============================================================
# # 5.5 拒绝推断
# ============================================================

# ============================================================
# # 5.6 案例1：某消费金融公司申请评分卡构建
# ============================================================

# ============================================================
# 5.6.1 场景介绍
# ============================================================

# ============================================================
# 5.6.2 数据清洗
# ============================================================

# ============================================================
# 1. 实体一致性
# ============================================================

# --- code cell ---
import pandas as pd
dat=pd.read_csv('creditdata_small.csv')
dat=dat.set_index('ID')
dat=dat.loc[:,~dat.columns.duplicated()] 
dat=dat.loc[~dat.index.duplicated(),:]

# ============================================================
# 2.列类型分配
# ============================================================

# --- code cell ---
dtypes_dict={
    'str':['GENDER', 'FLAG_OWN_CAR', 'HOUSING_TYPE', 'FLAG_OWN_REALTY',     'EDULEVEL','MARRIAGE'],
    'float':[i for i in dat.columns if i not in ['GENDER', 'FLAG_OWN_CAR', 'HOUSING_TYPE','FLAG_OWN_REALTY', 'EDULEVEL','MARRIAGE']]    
}     
for i in dtypes_dict['str']:
    dat[i]=dat[i].astype('str')
for i in dtypes_dict['float']:
    dat[i]=dat[i].astype('float') 

# ============================================================
# 3.数据质量报告
# ============================================================

# --- code cell ---
import toad
toad.detect(dat).T

# ============================================================
# 4.数据清洗
# ============================================================

# ============================================================
# （1）错误值处理
# ============================================================

# --- code cell ---
import numpy as np
dat['DAYS_EMPLOYED']=dat['DAYS_EMPLOYED'].map(lambda x:np.nan if x<0 else x)

# ============================================================
# （2）缺失值处理
# ============================================================

# --- code cell ---
missing_values_num=[np.nan,np.inf,-np.inf]
missing_values_obj=[np.nan,'nan','','special','missing']
dat=pd.concat(
[dat.select_dtypes(include='number').replace(missing_values_num,np.nan),           dat.select_dtypes(exclude='number').replace(missing_values_obj,'missing')],axis=1)

# --- code cell ---
dat.TARGET

# ============================================================
# 5.数据分区
# ============================================================

# --- code cell ---
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(dat.drop('TARGET',axis=1),dat.TARGET,test_size=0.33,random_state=1223,stratify=dat.TARGET)

# ============================================================
# 5.6.3 特征初筛
# ============================================================

# --- code cell ---
X_train_tmp1=toad.selection.drop_empty(X_train,threshold=0.9)
X_train_tmp2=toad.selection.drop_var(X_train_tmp1,threshold=0)
X_train_1=toad.selection.drop_iv(X_train_tmp1,y_train,threshold=0.02)
X_test_1=X_test[X_train_1.columns]

# ============================================================
# 5.6.4 分箱与WOE编码
# ============================================================

# ============================================================
# 1. 自动分箱
# ============================================================

# --- code cell ---
bins = toad.transform.Combiner()
bins.fit(X_train_1,y_train, method = 'dt',n_bins=5,min_samples = 0.05)

# ============================================================
# 以特征Age为例，其分箱切分点信息和分箱图
# ============================================================

# --- code cell ---
from toad.plot import  bin_plot,badrate_plot
bin_plot(bins.transform(X_train_1),x='Age',target=y_train)

# ============================================================
# 2.分箱调整
# ============================================================

# --- code cell ---
rule = {'Age':[36.5, 51.5, 57.5]}
bins.update(rule)
bin_plot(bins.transform(X_train_1,labels=True),x='Age',target=y_train)

# ============================================================
# 3.WOE编码
# ============================================================

# --- code cell ---
woe = toad.transform.WOETransformer()
X_train_woe = woe.fit_transform(bins.transform(X_train_1), y_train)
X_test_woe = woe.transform(bins.transform(X_test_1))

# ============================================================
# 5.6.5 相关性分析与特征聚类
# ============================================================

# --- code cell ---
X_train_woe_1=toad.selection.drop_corr(X_train_woe,y_train,threshold=0.7)
X_test_woe_1=X_test_woe[X_train_woe_1.columns]

# ============================================================
# 5.6.6 逐步回归
# ============================================================

# ============================================================
# 1. 逐步回归
# ============================================================

# --- code cell ---
X_train_woe_2 = toad.selection.stepwise(X_train_woe_1,y_train, 
	direction = 'both', criterion = 'aic')
X_test_woe_2 = X_test_woe_1[X_train_woe_2.columns]

# ============================================================
# 2.逻辑回归建模
# ============================================================

# --- code cell ---
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_woe_2,y_train)

# ============================================================
# 3.VIF诊断
# ============================================================

# --- code cell ---
toad.selection.VIF(X_train_woe_2)

# ============================================================
# 4.PSI
# ============================================================

# --- code cell ---
toad.metrics.PSI(X_train_woe_2, X_test_woe_2)

# ============================================================
# 5.6.7 模型评估
# ============================================================

# ============================================================
# 1.入模特征评估
# ============================================================

# ============================================================
# 2.样本内评估
# ============================================================

# --- code cell ---
import sklearn.metrics as metrics
fpr_train, tpr_train, _ = metrics.roc_curve(y_train, model.predict_proba(X_train_woe_2)[:,1])
fpr_test, tpr_test, _ = metrics.roc_curve(y_test, model.predict_proba(X_test_woe_2)[:,1])

# --- code cell ---
from matplotlib import pyplot as plt  
plt.plot(fpr_train,tpr_train,label = 'train LR:AUC = %0.2f'% metrics.auc(fpr_train, tpr_train)) 
plt.plot(fpr_test,tpr_test,label = 'test LR:AUC = %0.2f'% metrics.auc(fpr_test, tpr_test)) 
plt.plot([0,1],[0,1],'k--')  
plt.xlabel('False positive rate')  
plt.ylabel('True positive rate')  
plt.title('ROC Curve')  
plt.legend(loc = 'best') 

# --- code cell ---
th_tr=[i/len(tpr_train) for i in range(len(tpr_train))]
th_te=[i/len(tpr_test) for i in range(len(tpr_test))]
plt.plot(th_tr,tpr_train) 
plt.plot(th_tr,fpr_train)
plt.plot(th_te,tpr_test) 
plt.plot(th_te,fpr_test)
plt.plot(th_tr,abs(fpr_train-tpr_train),label = 'train ks = %.4f'%max(abs(fpr_train-tpr_train))) 
plt.plot(th_te,abs(fpr_test-tpr_test),label = 'test  ks = %.4f'%max(abs(fpr_test-tpr_test))) 
plt.xlabel('Population')  
plt.ylabel('Bad/Good Cumsum Population%')  
plt.title('KS Curve')  
plt.legend(loc = 'best') 

# ============================================================
# 5.6.8 评分卡制作
# ============================================================

# ============================================================
# 1. 刻度转换与分值分配
# ============================================================

# --- code cell ---
card = toad.ScoreCard(
    combiner = bins,
    transer = woe,
    base_score = 600,
    base_odds = 35,
    pdo = 50,
)
card.fit(X_train_woe_2,y_train)
sc=card.export(to_frame=True)
sc.head(10)

# --- code cell ---
X_train_woe_2['score']=card.predict(X_train_woe_2)
X_test_woe_2['score']=card.predict(X_test_woe_2)

# ============================================================
# 2. 评分表现报告
# ============================================================

# --- code cell ---
scorepf=toad.metrics.KS_bucket(X_train_woe_2['score'], y_train, bucket=10, method = 'quantile')

# --- code cell ---
print(scorepf)

# ============================================================
# 5.7 案例2：制作Vintage报告
# ============================================================

# ============================================================
# 5.7.1 Vintage相关业务报表
# ============================================================

# ============================================================
# 客户合同信息表
# ============================================================

# ============================================================
# 还款明细表内的一笔正常分期贷款的还款记录
# ============================================================

# ============================================================
# 逾期分期贷款的还款记录
# ============================================================

# ============================================================
# 5.7.2 Vintage报表的制作
# ============================================================

# ============================================================
# 1.使用Python3连接MySQL数据库
# ============================================================

# --- code cell ---
import pandas as pd
import pymysql
import sqlalchemy
query="""
show databases;
"""
risk = sqlalchemy.create_engine('mysql+pymysql://root:1122@localhost:3306/?charset=utf8mb4')
data=pd.read_sql(query,risk)
print(data)

# --- code cell ---
import pandas as pd
import pymysql
import sqlalchemy
query="""
select c.放款月,round(sum(放款金额/10000),2) 放款金额,
                           case when sum(放款金额)>0 then round(sum(mob1)/sum(放款金额),4) else null end as mob_1,
                           case when sum(放款金额)>0 then round(sum(mob2)/sum(放款金额),4) else null end as mob_2,
                           case when sum(放款金额)>0 then round(sum(mob3)/sum(放款金额),4) else null end as mob_3,
                           case when sum(放款金额)>0 then round(sum(mob4)/sum(放款金额),4) else null end as mob_4,
                           case when sum(放款金额)>0 then round(sum(mob5)/sum(放款金额),4) else null end as mob_5,
                           case when sum(放款金额)>0 then round(sum(mob6)/sum(放款金额),4) else null end as mob_6
                     from(
                           select 分期数,放款月,
                                  sum(case when mob=1 and 当前最大逾期天数>0 then 剩余本金 else 0 end) as mob1,
                                  sum(case when mob=2 and 当前最大逾期天数>0 then 剩余本金 else 0 end) as mob2,
                                  sum(case when mob=3 and 当前最大逾期天数>0 then 剩余本金 else 0 end) as mob3,
                                  sum(case when mob=4 and 当前最大逾期天数>0 then 剩余本金 else 0 end) as mob4,
                                  sum(case when mob=5 and 当前最大逾期天数>0 then 剩余本金 else 0 end) as mob5,
                                  sum(case when mob=6 and 当前最大逾期天数>0 then 剩余本金 else 0 end) as mob6
                           from ( 
                                  select m.分期数,m.剩余本金,m.放款月,m.观测月,m.当前最大逾期天数,
                                        case when substr(m.观测月,1,4)=substr(m.放款月,1,4) then substr(m.观测月,6,2)-substr(m.放款月,6,2)
                                             when substr(m.观测月,1,4)=substr(m.放款月,1,4)+1 then 12+substr(m.观测月,6,2)-substr(m.放款月,6,2)
                                        else 0 end as mob 
                                  from risk.repayment_sum_month m) a 
                          group by 分期数,放款月) b
                   join (select 合同期限,substr(放款日期,1,7) 放款月,
                                sum(合同金额) 放款金额,count(1) 放款量
                           from risk.customer_detail 
                          where 放款日期>='2017-11-01' 
                            and 放款日期<='2018-04-30'
                          group by 合同期限,substr(放款日期,1,7)) c 
                     on b.分期数=c.合同期限 and b.放款月=c.放款月
                  where 合同期限=6
                  group by c.放款月;
                  """
risk = sqlalchemy.create_engine('mysql+pymysql://root:1122@localhost:3306/?charset=utf8mb4')
data=pd.read_sql(query,risk)
print(data.sort_values('放款月'))

# --- code cell ---

