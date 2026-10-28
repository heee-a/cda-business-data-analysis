"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
import pandas as pd
import pymysql
import sqlalchemy

# --- code cell ---
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
                  group by c.放款月
                  """

# --- code cell ---
import pandas as pd
import pymysql
import sqlalchemy


risk = sqlalchemy.create_engine('mysql+pymysql://root:12345678@localhost:3306/?charset=utf8')
data=pd.read_sql(query,risk)

# --- code cell ---
data.sort_values('放款月')
