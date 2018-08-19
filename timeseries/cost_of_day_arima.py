'''
费用支出-按天统计
'''
from bokeh.io import output_notebook
from bokeh.plotting import figure
from pandas import read_csv
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

<<<<<<< HEAD

# init bokeh
output_notebook()


def plot_arima(truth, forecasts, title="ARIMA", xaxis_label='Time',
               yaxis_label='Value', c1='#A6CEE3', c2='#B2DF8A',
               forecast_start=None, **kwargs):
    # make truth and forecasts into pandas series
    n_truth = truth.shape[0]
    n_forecasts = forecasts.shape[0]

    # always plot truth the same
    truth = pd.Series(truth, index=np.arange(truth.shape[0]))

    # if no defined forecast start, start at the end
    if forecast_start is None:
        idx = np.arange(n_truth, n_truth + n_forecasts)
    else:
        idx = np.arange(forecast_start, n_forecasts)
    forecasts = pd.Series(forecasts, index=idx)

    # set up the plot
    p = figure(title=title, plot_height=400, **kwargs)
    p.grid.grid_line_alpha = 0.3
    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = yaxis_label

    # add the lines
    p.line(truth.index, truth.values, color=c1, legend='Observed')
    p.line(forecasts.index, forecasts.values, color=c2, legend='Forecasted')
    return p

data_input = read_csv('dataset/new_cost_of_month_weekday.csv')
data= data_input['group_fees']
=======
data_input= read_csv('dataset/workerCostByDay.csv', header=None)
data_input.columns=['date','total_fees','group_fees','hospital_fees','h_groupfees','menzhen_fees','m_groupfees','hospital_count','menzhen_count','avg_hgroupfees','avg_mgroupfees']
data=data_input['group_fees']
>>>>>>> 46046854230cec388d86f809dcce566558853598
###检测数据的稳定性
# data.plot()
# pyplot.show()
####进行趋势分解
# result = seasonal_decompose(data, model='multiplicative',freq=365)
# result.plot()
# pyplot.show()
####进行差分计算
###fitting a stepwise model:
size=-90
train=data[0:size]    ##训练集
test=data[size:-30]      ##测试集
stepwise_fit = auto_arima(train, start_p=1, start_q=1, max_p=2, max_q=2, m=12,
                          start_P=0, seasonal=True, d=1, D=1, trace=True,
                          error_action='ignore',  # don't want to know if an order does not work
                          suppress_warnings=True,  # don't want convergence warnings
                          stepwise=True)  # set to stepwise
print(stepwise_fit.summary())
<<<<<<< HEAD
# #
# # # #####预测未来几期的值
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
=======
#
# # #####预测未来几期的值
pre_value = stepwise_fit.predict(n_periods=len(test))
print(pre_value)
for i in range(1,len(pre_value)):
    print('预测值:%.3f'% pre_value[i])

test_mae = mean_absolute_percentage_error(test, pre_value)
print('Mean Absolute Percentage Error On Test Set: %.3f' % test_mae)
error_rate=error_rate_computed(test, pre_value)
print("测试误差率:",error_rate)
pyplot.plot(train)
pyplot.plot(test)
pyplot.plot(pre_value, color='red')
pyplot.show()
>>>>>>> 46046854230cec388d86f809dcce566558853598
