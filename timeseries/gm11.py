#-*- coding:utf-8 -*-
import string
import numpy as np
from numpy.linalg import inv
import pandas as pd
from sklearn.metrics import mean_absolute_error
###生成1-AGO累加序列
def AGO(X0):
    X1=[]
    for i in range(0,len(X0)):
        sum=0
        for j in range(0,i+1):
            sum=sum+X0[j]
        X1.append(sum)
    return X1

###光滑性检验

####生成矩阵B的一部分
def getB(X0):
    X1=[]
    for i in range(0,len(X0)-1):
        X1.append(-(0.5*X0[i]+0.5*X0[i+1]))
        X1.append(1)
    return X1

###对原始序列进行拟合
def fitted(k,X0,a,u):
    X1=[]
    X1.append(X0[0])
    for i in range(1,len(X0)):
        xi=round((1 - np.exp(a)) * (X0[0]-u /a) * np.exp(-a * i),0)
        X1.append(xi)
    return X1

###对未知序列进行预测
def predict(k,X0,a,u):
    result=round((1 - np.exp(a)) * (X0[0] - u / a) * np.exp(-a * (k - 1)),0)
    return result

######计算MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

####误差率
def error_rate_computed(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (y_true - y_pred) / y_true*100


###程序执行入口
# input_file=pd.read_csv('dataset/avgwage_working_retired.csv')
data=[65565,108845,148677,192740,226928,261060,332473, 403912, 414545,484936,657336,800100]
# data=input_file['avg_wage']
size=12
old_series=data[0:size]
new_series=AGO(old_series)

###建立gm11模型
###参数估计
Y=np.array(old_series[1:]).T
B=np.array(getB(new_series)).T.reshape(len(old_series)-1,2)
a=inv((B.T.dot(B))).dot(B.T).dot(Y)           ###参数估计
print('参数估计:',a)
###拟合原始序列
fit_value=fitted(len(old_series),old_series,a[0],a[1])
print("拟合值:",fit_value)
train_mae = mean_absolute_percentage_error(old_series, fit_value)
print('Mean Absolute Percentage Error On Train Set: %.3f' % train_mae)

###精度检验
avg_x=np.mean(old_series) ###原始序列的均值
error=np.array(fit_value)-np.array(old_series )###残差序列
avg_error=np.mean(error)  ###残差序列的均值
error_rate=error_rate_computed(old_series,fit_value)
print("拟合误差率:",error_rate)
##求原始数列标准差
S1= np.std(old_series)
##求残差数列标准差
S2=np.std(error)
##求标准差比
C=S2/S1
print("标准差比:%.3f"%C)
##求小误差率
count=0
for i in range(0,len(old_series)):
    if abs(error[i]-avg_error)<0.6745*S1:
        count=count+1
P=count/len(old_series)
print(P)


# ###对测试集进行评估
# nums=len(data)-size
# pre_value=[]
# for i in range(1,nums+1):
#     pre_value.append(predict(len(old_series)+i,old_series,a[0],a[1]))
# print("预测值:",pre_value)
# test_mae = mean_absolute_percentage_error(data[size:], pre_value)
# print('Mean Absolute Percentage Error On Test Set: %.3f' % test_mae)
# error_rate=error_rate_computed(data[size:], pre_value)
# print("测试误差率:",error_rate)


####采用滚动预测的方法
###对测试集进行评估
# old_series=old_series
# nums=len(data)-size
# pre_value=[]
# for i in range(1,nums+1):
#     new_series = AGO(old_series)
#     Y = np.array(old_series[1:]).T
#     B = np.array(getB(new_series)).T.reshape(len(new_series)-1,2)
#     a = inv((B.T.dot(B))).dot(B.T).dot(Y)  ###参数估计
#     result=predict(len(old_series)+1,old_series,a[0],a[1])   ###始终根据当前序列，估计下一个序列的值
#     old_series.append(result)
#     pre_value.append(result)
# print("预测值:",pre_value)
# test_mae = mean_absolute_percentage_error(data[size:], pre_value)
# print('Mean Absolute Percentage Error On Test Set: %.3f' % test_mae)
# error_rate=error_rate_computed(data[size:], pre_value)
# print("测试误差率:",error_rate)

####预测未来时间序列的值
nums=4
pre_value=[]
for i in range(1,nums+1):
    pre_value.append(predict(len(old_series)+i,old_series,a[0],a[1]))
print("未来4期预测值:",pre_value)

####采用滚动预测的方法
pre_value=[]
nums=4
for i in range(1,nums+1):
    new_series = AGO(old_series)
    Y = np.array(old_series[1:]).T
    B = np.array(getB(new_series)).T.reshape(len(new_series)-1,2)
    a = inv((B.T.dot(B))).dot(B.T).dot(Y)  ###参数估计
    result=predict(len(old_series)+1,old_series,a[0],a[1])   ###始终根据当前序列，估计下一个序列的值
    old_series.append(result)
    pre_value.append(result)
print("滚动预测值:",pre_value)
