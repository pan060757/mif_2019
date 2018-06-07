import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import numpy as np
from matplotlib import pyplot
######计算MAPE
from statsmodels.tsa.seasonal import seasonal_decompose


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

####误差率
def error_rate_computed(y_true, y_pred):
    return (y_true - y_pred) / y_true*100

input_file=pd.read_csv('dataset/avgwage_working_retired.csv')
data=input_file['retired']
size=8
train=data[0:size]
test=data[size:]
model=SimpleExpSmoothing(train).fit(smoothing_level=0.1)
pre_value = model.forecast(len(test))
print("简单指数平滑法-预测值:\n",pre_value)
test_mae = mean_absolute_percentage_error(test, pre_value)
print('Mean Absolute Percentage Error On Test Set: %.3f' % test_mae)
error_rate=error_rate_computed(test, pre_value)
print("测试误差率:",error_rate)

####Holt线性趋势预测法
model=Holt(train).fit(smoothing_level=0.1,smoothing_slope=0.05)
pre_value = model.forecast(len(test))
print("HoltWinter线性趋势-预测值:\n",pre_value)
test_mae = mean_absolute_percentage_error(test, pre_value)
print('Mean Absolute Percentage Error On Test Set: %.3f' % test_mae)
error_rate=error_rate_computed(test, pre_value)
print("测试误差率:",error_rate)


model=ExponentialSmoothing(train,trend='add').fit()

pre_value = model.forecast(len(test))
print("HoltWinter指数平滑法-预测值:\n",pre_value)
test_mae = mean_absolute_percentage_error(test, pre_value)
print('Mean Absolute Percentage Error On Test Set: %.3f' % test_mae)
error_rate=error_rate_computed(test, pre_value)
print("测试误差率:",error_rate)