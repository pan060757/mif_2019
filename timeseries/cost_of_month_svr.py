import time
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt

# convert series to supervised learning
from sklearn.svm import SVR


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

# load dataset
# 加载数据
data_input= read_csv('dataset/workerCostByMonth.csv', header=None)
#####
data_input.columns=['date','total_fees','group_fees','hospital_fees','h_groupfees','menzhen_fees','m_groupfees','hospital_count','menzhen_count','avg_hgroupfees','avg_mgroupfees']
#####只提取一个特征
values=data_input['group_fees']
values=values.reshape(-1,1)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_hours = 1
n_features = 1
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed)
print(reframed.shape)


# split into train and test sets
values = reframed.values
n_train_hours = -24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)


#####SVR回归建模
C_range =[0.001,0.01,0.1,1,10,100]
gamma_range = [1,2,3,4]
param_grid = dict(gamma=gamma_range, C=C_range)
cv = KFold(n_splits=5, shuffle=False, random_state=None)
svr = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, cv=cv)
# print(svr.best_params_)
##记录训练时间
svr.fit(train_X,train_y) ###拟合模型
yhat=svr.predict(test_X)

# invert scaling for forecast
yhat=yhat.reshape(len(yhat),1)
# inv_yhat = concatenate((yhat, test_X[:, -27:]), axis=1)
inv_yhat = scaler.inverse_transform(yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, -27:]), axis=1)
inv_y = scaler.inverse_transform(test_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = mean_absolute_percentage_error(inv_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_y,label='true_value',color='red')
pyplot.plot(inv_yhat,label='pre_value',color='blue')
pyplot.show()
