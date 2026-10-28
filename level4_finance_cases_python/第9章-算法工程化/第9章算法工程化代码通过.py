"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# # 第九章算法工程化代码
# ============================================================

# ============================================================
# # 9.1 构建合理的项目工程结构
# ============================================================

# ============================================================
# 9.1.1为什么要构建合理的项目工程结构
# ============================================================

# ============================================================
# 9.1.2什么是一个数据科学项目应有的项目结构
# ============================================================

# ============================================================
# # 9.2 如何编写规范的数据工程代码
# ============================================================

# ============================================================
# 9.2.1代码可读性
# ============================================================

# --- code cell ---
from sklearn.base import TransformerMixin
class CustomTimeTransformer(TransformerMixin):
    def __init__(self, time_col='timestamp', period_dict=None, 
                 season_dict=None):
        self.time_col = time_col
        self.period_dict = period_dict
        self.season_dict = season_dict

    def transform(self, df):
        data = df.copy()
        data[self.time_col] = data[self.time_col].apply(
            lambda x: pd.Timestamp(x))
        data['year'] = data[self.time_col].apply(lambda x: x.year)
        data['mth'] = data[self.time_col].apply(lambda x: x.month)
        data['day'] = data[self.time_col].apply(lambda x: x.day)
        data['hour'] = data[self.time_col].apply(lambda x: x.hour)
        data['minute'] = data[self.time_col].apply(lambda x: x.minute)
        data['second'] = data[self.time_col].apply(lambda x: x.second)
        data['minute_of_day'] = data[self.time_col].apply(
            lambda x: x.minute + x.hour * 60)
        data['week'] = data[self.time_col].apply(lambda x: x.dayofweek)
        data['day_of_year'] = data[self.time_col].apply(
            lambda x: x.dayofyear)
        data['week_of_year'] = data[self.time_col].apply(lambda x: x.week)
        
        if self.period_dict:
            data['period'] = data['hour'].map(self.period_dict)
        if self.season_dict:
            data['season'] = data['mth'].map(self.season_dict)
        return data

    def fit(self, *_):
        return self

class CustomTimeBooleanTransformer(TransformerMixin):
    def __init__(self, time_col='timestamp', public_vacations=[]):
        self.time_col = time_col
        self.public_vacations = public_vacations

    def transform(self, df):
        data = df.copy()
        data[self.time_col] = data[self.time_col].apply(
            lambda x: pd.Timestamp(x))
        data['is_leap_year'] = data[self.time_col].apply(
            lambda x: x.is_leap_year)
        data['is_mth_start'] = data[self.time_col].apply(
            lambda x: x.is_month_start)
        data['is_mth_end'] = data[self.time_col].apply(
            lambda x: x.is_month_end)
        data['is_quarter_start'] = data[self.time_col].apply(
            lambda x: x.is_quarter_start)
        data['is_quarter_end'] = data[self.time_col].apply(
            lambda x: x.is_quarter_end)
        data['is_year_start'] = data[self.time_col].apply(
            lambda x: x.is_year_start)
        data['is_year_end'] = data[self.time_col].apply(
            lambda x: x.is_year_end)
        data['is_weekend'] = data[self.time_col].apply(
            lambda x: True if x.dayofweek in [5, 6] else False)

        data['is_working'] = False
        data['tmp_hour'] = data[self.time_col].apply(lambda x: x.hour)
        data.loc[((data['tmp_hour'] >= 8) & (data['tmp_hour'] < 22)),
                 'is_working'] = True
        data.drop(['tmp_hour'], axis=1, inplace=True)

        if self.public_vacations:
            data['is_public_holiday'] = data[self.time_col].apply(
                lambda x: True if x.strftime('%Y%m%d') in \
                self.public_vacations else False)
        return data

    def fit(self, *_):
        return self

class CustomTimeSeriesTransformer(TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def transform(self, df):
        data = df.copy()
        for col in self.cols:
            # 时间序列历史特征
            for i in range(1, 4):
                data["{}-{}day".format(col, i)] = data[col].shift(1)

            # 趋势特征
            data[col + '_last_3_day_mean'] = \
            (data[col].shift(1)+ data[col].shift(2) + data[col].shift(3))/ 3
            # 窗口差异值特征
            data[col + '_last_2_mins'] = \
            data[col].shift(1) - data[col].shift(2)
            # 自相关特征
            data[col + '_lag_1_corr'] = data[col].autocorr(1)
            data[col + '_lag_2_corr'] = data[col].autocorr(2)
        return data

    def fit(self, *_):
        return self

# ============================================================
# 当涉及多个数据处理任务时，可以并行开发
# ============================================================

# --- code cell ---
import pandas as pd
# 创建测试数据
data = {
    'timestamp': ['2023-01-01 10:00:00', '2023-02-01 15:30:00'],
    'col1': [10, 20]
}
df = pd.DataFrame(data)


# 调用CustomTimeTransformer
time_transformer = CustomTimeTransformer(
    time_col='timestamp',
    period_dict={0: 'off-peak', 6: 'peak', 12: 'off-peak', 18: 'peak'},
    season_dict={1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'summer',
    6: 'summer', 7: 'summer', 8: 'autumn', 9: 'autumn', 10: 'autumn',
    11: 'winter', 12: 'winter'}
)
transformed_time_data = time_transformer.transform(df)
print("CustomTimeTransformer转换后的数据：")
print(transformed_time_data.head())

# 调用CustomTimeBooleanTransformer
boolean_transformer = CustomTimeBooleanTransformer(
    time_col='timestamp',
    public_vacations=['20230101']
)
transformed_boolean_data = boolean_transformer.transform(df)
print("\nCustomTimeBooleanTransformer转换后的数据：")
print(transformed_boolean_data.head())

# 调用CustomTimeSeriesTransformer
series_transformer = CustomTimeSeriesTransformer(cols=['col1'])
transformed_series_data = series_transformer.transform(df)
print("\nCustomTimeSeriesTransformer转换后的数据：")
print(transformed_series_data.head())

# ============================================================
# 9.2.2数据处理性能
# ============================================================

# --- code cell ---
'''
1.使用HDF, Feather, h5py 格式文件加快数据读取
2.使用 itertuples, iterrows 代替for循环
3.使用apply方法替代 iterrows
4.使用 apply, transform, agg 函数时尽量使用内置函数
5.类型参数优化
'''

# --- code cell ---

