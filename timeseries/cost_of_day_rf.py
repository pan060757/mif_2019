import time
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
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
data_input = read_csv('dataset/new_cost_of_month_weekday.csv')
data_input = data_input[
    ['weekday', 'month', 'season', 'weekday_or_not', 'holiday_or_not', 'h_count', 'h_groupfees', 'm_count',
     'm_groupfees', 'group_fees']]
# ensure all data is float
values = data_input.astype('float32')

###进行哑变量处理(1+12+7+4+2+2=28)
##对个别列进行哑变量处理(7个星期特征)
weekday_dummy = pd.get_dummies(values.weekday, prefix='weekday')
values = values.join(weekday_dummy)
#####12个月份特征
month_dummy = pd.get_dummies(values.month, prefix='month')
values = values.join(month_dummy)
#######4个季度特征
season_dummy = pd.get_dummies(values.season, prefix='season')
values = values.join(season_dummy)
#######2个是否工作日特征
weekday_or_dummy = pd.get_dummies(values.weekday_or_not, prefix='weekday_or_not')
values = values.join(weekday_or_dummy)
#######2个是否节假日特征
holiday_or_dummy = pd.get_dummies(values.holiday_or_not, prefix='holiday_or_dummy')
values = values.join(holiday_or_dummy)

######剔除原始的特征
values = values.drop(
    ['group_fees', 'month', 'weekday', 'season', 'weekday_or_not', 'holiday_or_not', 'h_count', 'h_groupfees',
     'm_count', 'm_groupfees'], 1)
values = values.join(
    [data_input['h_count'], data_input['h_groupfees'], data_input['m_count'], data_input['m_groupfees'],
     data_input['group_fees']])

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# specify the number of lag hours
n_hours = 2
n_features = 91
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed)
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_hours = -90
train = values[:n_train_hours, :]
test = values[n_train_hours:-30, :]
# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :-5], train[:, -1]
test_X, test_y = test[:, :-5], test[:, -1]
print(train_X.shape, len(train_X), train_y.shape)

#####RandomForest
rf=RandomForestRegressor()
parameters = {'n_estimators': [100,200,300,400,500], 'max_features':[4,5,6,7,8,9,10]}
grid_search = GridSearchCV(estimator=rf,param_grid=parameters, cv=5)
grid_search.fit(train_X,train_y)
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters=grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

####对模型进行打分
preds_train = grid_search.predict(train_X)
print("模型打分情况：", metrics.r2_score(train_y, preds_train))

yhat=grid_search.predict(test_X)
# invert scaling for forecast
yhat=yhat.reshape(len(yhat),1)
inv_yhat = concatenate((test_X[:, -31:],yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, -31:],test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]
# calculate RMSE
rmse = mean_absolute_percentage_error(inv_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_y,label='true_value',color='red')
pyplot.plot(inv_yhat,label='pre_value',color='blue')
pyplot.show()
