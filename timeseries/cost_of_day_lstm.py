import time

import keras
from keras import Input
from keras import Model
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

#####转为有监督的学习
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
####读入介入星期特征以及月份特征的数据
data_input= read_csv('dataset/new_cost_of_month_weekday.csv')
values=data_input[['weekday','month','season','weekday_or_not','holiday_or_not','h_groupfees','h_count','group_fees']]
# ensure all data is float
values = values.astype('float32')


##对个别列进行哑变量处理(7个星期特征)
weekday_dummy=pd.get_dummies(values.weekday,prefix='weekday')
values=values.join(weekday_dummy)
#####12个月份特征
month_dummy=pd.get_dummies(values.month,prefix='month')
values=values.join(month_dummy)
#######4个季度特征
season_dummy=pd.get_dummies(values.season,prefix='season')
values=values.join(season_dummy)
#######2个是否工作日特征
weekday_or_dummy=pd.get_dummies(values.weekday_or_not,prefix='weekday_or_not')
values=values.join(weekday_or_dummy)
#######2个是否节假日特征
holiday_or_dummy=pd.get_dummies(values.holiday_or_not,prefix='holiday_or_dummy')
values=values.join(holiday_or_dummy)

######剔除原始的特征
values = values.drop(['group_fees', 'month', 'weekday', 'season', 'weekday_or_not', 'holiday_or_not','h_count','h_groupfees'], 1)
values = values.join([data_input['h_count'],data_input['h_groupfees'],data_input['group_fees']])
print(values)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_hours = 8
n_features = 30
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed)
print(reframed.shape)
values = reframed.values

# split into train and test sets
n_train_hours = -90
train = values[:n_train_hours, :]
test = values[n_train_hours:-30, :]
# split into input and outputs
n_obs = n_hours * n_features
print(train[:, :n_obs])
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


######使用LSTM模型
#####求10次实验的平均值
# repeats =10
# error_scores = list()
###design network
# for r in range(repeats):
# model = Sequential()
# model.add(LSTM(12, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))

# 为模型添加额外的信息
auxiliary_train=train[:, n_obs:-3]
auxiliary_train=auxiliary_train.reshape(auxiliary_train.shape[0],1,auxiliary_train.shape[1])
auxiliary_test=test[:, n_obs:-3]
auxiliary_test=auxiliary_test.reshape(auxiliary_test.shape[0],1,auxiliary_test.shape[1])
main_input=Input(shape=(train_X.shape[1], train_X.shape[2]),name='main_input')
lstm_out=LSTM(20,dropout=0.3)(main_input)
auxiliary_output=Dense(1,name='aux_output')(lstm_out)
auxiliary_input=Input(shape=(auxiliary_train.shape[1],auxiliary_train.shape[2]),name='aux_input')
merge_out=keras.layers.concatenate([lstm_out,auxiliary_output])
main_output=Dense(1)(merge_out)
model=Model(input=[main_input,auxiliary_input],outputs=main_output)
model.compile(loss='mae', optimizer='adam')
print(model.summary())
# fit network
history=model.fit([np.array(train_X),np.array(auxiliary_train)],np.array(train_y),epochs=100, batch_size=72,
                  validation_data=([test_X,auxiliary_test], test_y),verbose=2)
# history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2,
#                     shuffle=False)
# plot history
# pyplot.plot(history.history['loss'], label='train',color='red')
# pyplot.plot(history.history['val_loss'], label='test',color='blue')
# pyplot.legend()
# pyplot.show()


# make a prediction
yhat = model.predict([test_X,auxiliary_test])
test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -29:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -29:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = mean_absolute_percentage_error(inv_y, inv_yhat)
####将结果写入文件
out=open("dataset/pre_true.csv","w+")
for i in range(len(inv_y)):
    out.write(str(inv_y[i])+','+str(inv_yhat[i])+'\n')
out.close()
print('Test RMSE: %.3f' % rmse)
######
pyplot.plot(inv_y,label='true_value',color='red')
pyplot.plot(inv_yhat,label='pre_value',color='blue')
pyplot.show()


