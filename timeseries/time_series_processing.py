#coding:utf-8
'''
时间序列数据预处理
'''
from keras import Sequential
from keras.layers import LSTM, Dense
from numpy.ma import concatenate
from pandas import concat, DataFrame, read_csv
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib import pyplot
import numpy as np

def series_to_supervised(data, n_in=1, n_history=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    ####加入历史同期数据
    if n_history>0:
        for i in range(n_history, 0, -1):
            cols.append(df.shift(i*7))
            names += [('var%d(t-%d)' % (j + 1, i*7)) for j in range(n_vars)]

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

######生成多变量的数据集
def generate_mul_dataset(n_in,n_history,n_out, n_train_sample):
    data_input = read_csv('dataset/new_cost_of_month_trend.csv')
    data_input = data_input[
        ['weekday', 'month', 'season', 'weekday_or_not', 'holiday_or_not','h_count', 'h_groupfees', 'm_count',
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
    # # #######上升还是下降趋势
    # trend_dummy = pd.get_dummies(values.trend, prefix='trend')
    # values = values.join(trend_dummy)
    #####剔除原始的特征
    values = values.drop( ['group_fees', 'month', 'weekday', 'season', 'weekday_or_not', 'holiday_or_not', 'h_count', 'h_groupfees',
         'm_count', 'm_groupfees'], 1)
    values = values.join(
        [data_input['h_count'], data_input['h_groupfees'], data_input['m_count'], data_input['m_groupfees'],
         data_input['group_fees']])


    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify the number of lag hours
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_in, n_out)
    print(reframed.shape)
    # split into train and test sets
    values = reframed.values
    train = values[:n_train_sample, :]
    test = values[n_train_sample:-30, :]
    # split into input and outputs
    train_X, train_y = train[:, :-5], train[:, -1]
    test_X, test_y = test[:, :-5], test[:, -1]
    print(train_X.shape, len(train_X), train_y.shape)
    return scaler,train_X,train_y,test_X,test_y

#####生成单变量的数据集
def generate_uni_dataset(n_in,n_history,n_out,n_train_sample):
    ####读入介入星期特征以及月份特征的数据
    data_input = read_csv('dataset/new_cost_of_month_weekday.csv')
    ###############单变量数据建模######################
    data_input = data_input['group_fees']
    values = data_input.values
    values = values.reshape(-1, 1)
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify the number of lag hours
    # frame as supervised learning
    reframed = series_to_supervised(scaled,n_in, n_history, n_out)
    print(reframed.shape)
    # split into train and test sets
    values = reframed.values
    train = values[:n_train_sample, :]
    test = values[n_train_sample:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    print(train_X.shape, len(train_X), train_y.shape)
    return scaler,train_X, train_y, test_X, test_y

####计算MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

###对多变量结果进行展示
def plot_mul_results(scaler,test_X,yhat,test_y):
    # invert scaling for forecast
    yhat = yhat.reshape(len(yhat), 1)
    inv_yhat = concatenate((test_X[:, -31:],yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate(( test_X[:, -31:],test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]
    # calculate RMSE
    rmse = mean_absolute_percentage_error(inv_y, inv_yhat)
    print('Test RMSE: %.3f' % rmse)
    ####将结果写入文件
    out = open("dataset/pre_true.csv", "w+")
    for i in range(len(inv_y)):
        out.write(str(inv_y[i]) + ',' + str(inv_yhat[i]) + '\n')
    out.close()
    pyplot.plot(inv_y, label='true_value', color='red')
    pyplot.plot(inv_yhat, label='pre_value', color='blue')
    pyplot.show()

####进行单变量结果展示
def plot_uni_results(scaler,yhat,test_y):
    yhat = yhat.reshape(-1, 1)
    inv_yhat = scaler.inverse_transform(yhat)
    # invert scaling for actual
    test_y = test_y.reshape(-1, 1)
    inv_y = scaler.inverse_transform(test_y)
    # calculate RMSE
    rmse = mean_absolute_percentage_error(inv_y, inv_yhat)
    print('Test RMSE: %.3f' % rmse)
    pyplot.plot(inv_y, label='true_value', color='red')
    pyplot.plot(inv_yhat, label='pre_value', color='blue')
    pyplot.show()

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

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted

####进行特征选择的过程,返回进行特征选择之后的结果
def selectFeatures(train_X,test_X,model_result):
    model_result=SelectFromModel(model_result,prefit=True)
    train_feature=model_result.transform(train_X)
    train_feature2=model_result.transform(test_X)
    return train_feature,train_feature2