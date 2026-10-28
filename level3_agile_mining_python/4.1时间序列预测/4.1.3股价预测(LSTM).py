"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# ============================================================
# ## 读取 Google 股价
# ============================================================

# --- code cell ---
import pandas as pd
googl = pd.read_csv('google_stock_prics.csv')
googl.head()

# ============================================================
# ## 将数据分为训练与测试数据集
# ============================================================

# --- code cell ---
data_length = len(googl)                 # 序列长度
train_size = 0.8                         # 训练集比例
cutoff = round(data_length * train_size) # 序列切分点

# 选择`Open`, `Close` 两列，划分训练集测试集
train = googl.iloc[:cutoff, 3:5]
test = googl.iloc[cutoff:, 3:5]

print("Train shape:", train.shape, "\nTest shape:", test.shape)

# ============================================================
# ## 资料标准化
# ============================================================

# --- code cell ---
from sklearn.preprocessing import MinMaxScaler

# 训练集极差标准化
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)

# 测试集标准化用训练集的极差
test_scaled = scaler.transform(test)  

# --- code cell ---
import numpy as np

def process_ts(ts, timesteps=4):
    """
    ts: numpy.array 2D 
    timesteps: 用于预测的序列步长
    returns: `x` with shape [n_samples, timesteps, features]
             `y` with shape [n_samples, features]
    """
    feature_dims = ts.shape[1]
    x, y = [], []
    for i in range(timesteps, len(ts)):
        x.append(ts[i - timesteps: i,])
        y.append(ts[i:i + 1, ])
        
    # 堆叠列表中的array   
    x, y = np.vstack(x), np.vstack(y)  
    # x转换维度为[n_samples, timesteps, features]
    x = x.reshape((-1, timesteps, feature_dims))
    # y转换维度为[samples, features]
    y = y.reshape((-1, feature_dims))
    return x, y

# --- code cell ---
timesteps = 60

train_data, train_target = process_ts(train_scaled, timesteps=timesteps)
test_data, test_target = process_ts(test_scaled, timesteps=timesteps)
print(train_data[:2])
print("-" * 20)
print(train_target[:2])

# ============================================================
# ## 建构LSTM神经网路
# ============================================================

# --- code cell ---
import tensorflow as tf
print(tf.__version__)

# --- code cell ---
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

feature_dims = train.shape[1]

regressor = Sequential()
regressor.add(LSTM(units=32, return_sequences=True, 
                   input_shape=(timesteps, feature_dims)))
regressor.add(LSTM(units=32, return_sequences=False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=feature_dims, activation=None))

regressor.summary()

# ============================================================
# ## 训练LSTM
# ============================================================

# --- code cell ---
# 选用MSE损失，adam优化器，编译模型
regressor.compile(optimizer='adam', loss='mean_squared_error')

# 训练
history = regressor.fit(
    train_data,             # 训练集x
    train_target,           # 训练集y
    epochs=100,             # 全量数据训练一次为一个epoch
    batch_size=16,          # 批量梯度下降中的批大小
    validation_data=(test_data, test_target),   # 验证集
    shuffle=False
)

# ============================================================
# ## 评估模型在测试数据集的表现
# ============================================================

# --- code cell ---
import matplotlib.pyplot as plt

fig = plt.figure(figsize=[10, 5])
ax = plt.subplot()
ax.plot(history.history['loss'], 'b-', label='train')
ax.plot(history.history['val_loss'], 'r--', label='test')
# ax.set_yscale('log')   # 纵轴使用对数坐标
plt.title('Change in Loss ', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.xlabel('epoch', fontsize=15)
plt.legend(fontsize=15)
plt.show()

# --- code cell ---
from sklearn.metrics import mean_absolute_error

# 预测 -> "反标准化"
test_predict = regressor.predict(test_data)
test_predict = scaler.inverse_transform(test_predict)
# 评估平均绝对误差
test_mae = mean_absolute_error(test_predict[:, 0], test.iloc[timesteps:, -1])

# 选定基线模型：使用前一天的收盘价作为第二天收盘价的预测，评估MSE
dummy_mae = mean_absolute_error(test.iloc[timesteps:-1, -1], test.iloc[timesteps + 1:, -1])
print("Mean absolute error : %.2f" %(test_mae))
print("Dummy MSE: %.2f" %(dummy_mae))

# --- code cell ---
plt.figure(figsize=(10, 5))
plt.plot(test.values[timesteps:, 1], "b-", label='Google Stock Price')
plt.plot(test_predict[:, 0],"r--", label='Predict Price')
plt.title('Google Stock Price Prediction', fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Google Stock Price', fontsize=15)
plt.legend(fontsize=15)
plt.show()

# --- code cell ---


# --- code cell ---

