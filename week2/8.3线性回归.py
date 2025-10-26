#导入库
import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split #划分训练集和测试集
from sklearn.linear_model import LinearRegression #线性回归模型函数
from  sklearn.metrics import mean_squared_error, r2_score

#加载数据集
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

#提前数据预处理，转换为DataFrame形式
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target
print(f"数据集的前五行：{df.head()}")
df.info()

X = df.drop('Price',axis=1) #移除列(axis=0移除行)，剩下的均为特征(因素)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#创建和训练模型
model = LinearRegression() #fit函数使用的是最小二乘法，可以帮助找到最优W,b
model.fit(X_train,y_train)
print(f'模型的系数和截距：{model.coef_},{model.intercept_}')

#使用模型进行预测
y_pred = model.predict(X_test)
print(X_test)
print(X_train)
print(y_test)
print(y_train)
print(y_pred)

#对模型进行评估
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f'均方误差和R2分数：{mse:.2f}，{r2:.2f}')