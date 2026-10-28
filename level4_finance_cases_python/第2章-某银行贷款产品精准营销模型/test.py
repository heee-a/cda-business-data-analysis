"""Converted from a CDA textbook Jupyter notebook.
原作者的讲解以 # 注释形式保留，便于理解代码内涵。
"""


# --- code cell ---
import pandas as pd
# 读取数据
snd =pd.read_csv('C:/Users/wfyh/data12261.csv')
# 数据形状
snd.shape

# --- code cell ---
snd.fillna(0,inplace = True)
del snd['account_id']
del snd['amount']

# --- code cell ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. 数据预处理
def preprocess_data(df):
    # 处理缺失值
    df = df.dropna()
    
    # 处理异常值
    def remove_outliers(df, cols):
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df
    
    numerical_cols = df.drop(columns = ['y']).columns
    df = remove_outliers(df, numerical_cols)
    print(df)
    return df


# 2. 特征选择
def select_features(X, y, n_features=5):
    logistic = LogisticRegression(random_state=42)
    rfe = RFE(estimator=logistic, n_features_to_select=n_features)
    rfe = rfe.fit(X, y)
    return rfe.support_

# 3. 模型训练和评估
def build_model(df):
    # 数据预处理
    df = preprocess_data(df)
    
    # 分割特征和目标变量
    X = df.drop(['y'], axis=1)
    y = df['y']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5,  stratify=y
    )
    
    # 标准化数值特征
    scaler = StandardScaler()
    numerical_cols = df.drop(columns = ['y']).columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 特征选择
    selected_features = select_features(X_train, y_train)
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    # 添加常数项
    X_train_sm = sm.add_constant(X_train_selected)
    X_test_sm = sm.add_constant(X_test_selected)
    
    # 训练模型
    model = sm.Logit(y_train, X_train_sm)
    results = model.fit()
    # 模型评估
    y_pred_train = results.predict(X_train_sm)
    y_pred_test = results.predict(X_test_sm)
    
    print("模型摘要：")
    print(results.summary())
    
    print("\n训练集评估指标：")
    print(classification_report(y_train, (y_pred_train > 0.5).astype(int)))
    print(f"AUC-ROC: {roc_auc_score(y_train, y_pred_train):.4f}")
    
    print("\n测试集评估指标：")
    print(classification_report(y_test, (y_pred_test > 0.5).astype(int)))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_test):.4f}")
    
    return results, X_train_selected.columns

# 运行模型
model_results, selected_features = build_model(snd)
 

# --- code cell ---
build_model(snd)

# --- code cell ---

