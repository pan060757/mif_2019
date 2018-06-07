from pyramid.arima import auto_arima
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas import read_csv, Series
from pandas import DataFrame
# load dataset
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

####误差率
def error_rate_computed(y_true, y_pred):
    return (y_true - y_pred) / y_true*100

data_input= read_csv('dataset/workerCostByMonth.csv', header=None)
data_input.columns=['date','total_fees','group_fees','hospital_fees','h_groupfees','menzhen_fees','m_groupfees','hospital_count','menzhen_count','avg_hgroupfees','avg_mgroupfees']
data=data_input['group_fees']
###检测数据的稳定性
# data.plot()
# pyplot.show()
####进行趋势分解
# result = seasonal_decompose(data, model='multiplicative',freq=12)
# result.plot()
# pyplot.show()

####进行差分计算
# fitting a stepwise model:
size=96
train=data[0:size]    ##训练集
test=data[size:]      ##测试集
# stepwise_fit = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
#                           start_P=0, seasonal=True, d=1, D=1, trace=True,
#                           error_action='ignore',  # don't want to know if an order does not work
#                           suppress_warnings=True,  # don't want convergence warnings
#                           stepwise=True)  # set to stepwise
# print(stepwise_fit.summary())
#
# #####预测未来几期的值
# pre_value = stepwise_fit.predict(n_periods=len(test))
# print(pre_value)
# for i in range(1,len(pre_value)):
#     print('预测值:%.3f'% pre_value[i])
#
# test_mae = mean_absolute_percentage_error(test, pre_value)
# print('Mean Absolute Percentage Error On Test Set: %.3f' % test_mae)
# error_rate=error_rate_computed(test, pre_value)
# print("测试误差率:",error_rate)
# pyplot.plot(train)
# pyplot.plot(test)
# pyplot.plot(pre_value, color='red')
# pyplot.show()
#
#
# #####更新模型
# print("更新模型：")
# updated_data = np.concatenate([train, pre_value])
# updated_model = stepwise_fit.fit(updated_data)
# print(updated_model.summary())

# model=SARIMAX(train,order=(0,1,1),seasonal_order=(0,1,2,12)).fit()
# pre_value=model.predict(len(test))
# for i in range(1,len(pre_value)):
#     print('预测值:%.3f'% pre_value[i])
#
# test_mae = mean_absolute_percentage_error(test, pre_value)
# print('Mean Absolute Percentage Error On Test Set: %.3f' % test_mae)
# error_rate=error_rate_computed(test, pre_value)
# print("测试误差率:",error_rate)
# pyplot.plot(train)
# pyplot.plot(test)
# pyplot.plot(pre_value, color='red')
# pyplot.show()

####HoltWinter
model=ExponentialSmoothing(train).fit(smoothing_level=1,smoothing_slope=0.3,smoothing_seasonal=0.4)
pre_value=model.forecast(len(test))
print('预测值', pre_value)

test_mae = mean_absolute_percentage_error(test, pre_value)
print('Mean Absolute Percentage Error On Test Set: %.3f' % test_mae)
error_rate=error_rate_computed(test, pre_value)
print("测试误差率:",error_rate)
pyplot.plot(train)
pyplot.plot(test)
pyplot.plot(pre_value, color='red')
pyplot.show()