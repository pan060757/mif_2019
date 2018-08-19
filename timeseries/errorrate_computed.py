#-coding:utf-8-*-
'''
误差率计算
'''
import pandas as pd
import numpy as np
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

data_input=pd.read_csv("dataset/cost_of_month_pre_true.csv")
true_value=data_input['true_value']
pre_value=data_input['pre_value']
print(mean_absolute_percentage_error(true_value,pre_value))
