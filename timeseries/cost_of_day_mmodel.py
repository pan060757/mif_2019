#-*-coding:utf-8-*-
'''
混合模型建立（xgbt+gbdt）
'''
from matplotlib import pyplot
from numpy import loadtxt, sort, concatenate
from pandas import concat
from pandas import read_csv
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from xgboost import plot_importance
from xgboost import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def series_to_supervised(data, n_in=1,n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    ###加入历史同期数据
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


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# load dataset
####读入介入星期特征以及月份特征的数据
if __name__=='__main__':
    data_input = read_csv('dataset/new_cost_of_month_weekday.csv')
    data_input = data_input[['weekday', 'month', 'season', 'weekday_or_not', 'holiday_or_not','h_count','h_groupfees','m_count','m_groupfees','group_fees' ]]
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
    values = values.drop(['group_fees', 'month', 'weekday', 'season', 'weekday_or_not', 'holiday_or_not','h_count','h_groupfees','m_count','m_groupfees'], 1)
    values = values.join([data_input['h_count'], data_input['h_groupfees'],data_input['m_count'], data_input['m_groupfees'],data_input['group_fees']])


    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify the number of lag hours
    n_hours = 2
    ####加入历史同期的数据
    n_history=3      ####前3个星期同期数据
    n_out=1          ####单输出
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours,n_out)
    print(reframed)
    print(reframed.shape)

    # split into train and test sets
    values = reframed.values
    n_train_hours = -90
    train = values[:n_train_hours, :]
    test = values[n_train_hours:-30, :]
    # split into input and outputs
    train_X, train_y = train[:, :-5], train[:, -1]
    test_X, test_y = test[:, :-5], test[:, -1]
    print(train_X.shape, len(train_X), train_y.shape)

    ####################xgbt模型###############################
    eval_set = [(test_X, test_y)]
    xgbt_model = XGBRegressor(learning_rate=0.05,n_estimators=90,max_depth=4)
    xgbt_model.fit(train_X, train_y,eval_metric=['rmse'],eval_set=eval_set,verbose=True)
    # feature importance
    # print(xgbt_model.feature_importances_)
    ####对模型进行打分
    xgbt_train =xgbt_model.predict(train_X)
    print("xgbt-模型打分情况：",metrics.r2_score(train_y,xgbt_train))
    xgbt_y = xgbt_model.predict(test_X)
    # invert scaling for forecast
    xgbt_yhat = xgbt_y.reshape(len(xgbt_y), 1)
    inv_yhat = concatenate((test_X[:, -31:],xgbt_yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate(( test_X[:, -31:],test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]
    # calculate RMSE
    rmse = mean_absolute_percentage_error(inv_y, inv_yhat)
    print('xgbt1-Test MAPE: %.3f' % rmse)

    ####################gbdt模型###############################
    param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
    gbdt_model = GridSearchCV(
        estimator=GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, min_samples_split=250,
                                            min_samples_leaf=20, max_depth=9, max_features='sqrt', subsample=0.8,
                                            random_state=10), param_grid=param_test2, iid=False, cv=5)
    gbdt_model.fit(train_X, train_y)
    print(gbdt_model.grid_scores_)
    print(gbdt_model.best_params_)
    gbdt_train = gbdt_model.predict(train_X)
    print("gbdt-模型打分情况：", metrics.r2_score(train_y, gbdt_train))
    gbdt_y = gbdt_model.predict(test_X)
    # invert scaling for forecast
    gbdt_yhat = gbdt_y.reshape(len(gbdt_y), 1)
    inv_yhat = concatenate((test_X[:, -31:],gbdt_yhat), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate(( test_X[:, -31:],test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]
    # calculate RMSE
    mape = mean_absolute_percentage_error(inv_y, inv_yhat)
    print('gbdt-Test MAPE: %.3f' % mape)


    #####利用前两次建模的结果作为此次输入
    new_train_x=pd.concat([pd.Series(xgbt_train),pd.Series(gbdt_train)],axis=1)
    #####再次进行xgbt模型
    model = XGBRegressor(learning_rate=0.05, n_estimators=90, max_depth=4)
    model.fit(new_train_x, train_y)
    ####对模型进行打分
    xgbt2_train = model.predict(new_train_x)
    print("xgbt-模型打分情况：", metrics.r2_score(train_y, xgbt2_train))
    new_test_x = pd.concat([pd.Series(xgbt_y), pd.Series(gbdt_y)], axis=1)
    yhat = model.predict(new_test_x)
    # invert scaling for forecast
    yhat = yhat.reshape(len(yhat), 1)
    inv_yhat = concatenate((test_X[:, -31:], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_X[:, -31:], test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]
    # calculate RMSE
    mape = mean_absolute_percentage_error(inv_y, inv_yhat)
    print('xgbt-Test MAPE: %.3f' % mape)





    # ####将结果写入文件
    # out = open("dataset/pre_true.csv", "w+")
    # for i in range(len(inv_y)):
    #     out.write(str(inv_y[i]) + ',' + str(inv_yhat[i]) + '\n')
    # out.close()
    # pyplot.plot(inv_y, label='true_value', color='red')
    # pyplot.plot(inv_yhat, label='pre_value', color='blue')
    # pyplot.show()
