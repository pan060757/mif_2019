# plot decision tree
from matplotlib import pyplot
from numpy import loadtxt, sort, concatenate
from pandas import concat
from pandas import read_csv
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
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
    n_features = 91  ####每个t时刻对应32个特征（再加上t+1时刻的27个特征）
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

    eval_set=[(test_X,test_y)]
    # fit model no training data
    ####learning_rate={0.01,0.05,0.1,0.2}
    ####max_math={3,4,5}
    ####n_estimators={80,90,100,110,120}
    model = MLPRegressor()
    model.fit(train_X,train_y)
    print(model.get_params())
    ####对模型进行打分
    preds_train =model.predict(train_X)
    print("模型打分情况：",model.score(train_X,train_y))

    yhat = model.predict(test_X)
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

    # retrieve performance metrics
    # results = model.evals_result()
    # print(results)
    # epochs = len(results['validation_0']['rmse'])
    # x_axis = range(0, epochs)
    # # plot log loss
    # fig, ax = pyplot.subplots()
    # ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    # ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    # ax.legend()
    # pyplot.ylabel('rsme')
    # pyplot.title('XGBoost Log Loss')
    # pyplot.show()
    # plot classification error
    # fig, ax = pyplot.subplots()
    # ax.plot(x_axis, results['validation_0']['mae'], label='Train')
    # ax.plot(x_axis, results['validation_1']['mae'], label='Test')
    # ax.legend()
    # pyplot.ylabel('mae')
    # pyplot.title('XGBoost Classification Error')
    # pyplot.show()


    # # plot single tree
    # plot_tree(model,num_trees=0,rankdir='LR')   ####可以刻画树模型
    # plt.show()
    # Fit model using each importance as a threshold
    # thresholds = sort(model.feature_importances_)
    # for thresh in thresholds:
    # # select features using threshold
    #         selection = SelectFromModel(model, threshold=thresh, prefit=True)
    #         select_X_train = selection.transform(train_X)
    #         # train model
    #         selection_model = XGBRegressor()
    #         selection_model.fit(select_X_train, train_y)
    #         # eval model
    #         select_X_test = selection.transform(test_X)
    #         yhat = selection_model.predict(select_X_test)
    #         # invert scaling for forecast
    #         yhat = yhat.reshape(len(yhat), 1)
    #         inv_yhat = concatenate((yhat, test_X[:, -29:]), axis=1)
    #         inv_yhat = scaler.inverse_transform(inv_yhat)
    #         inv_yhat = inv_yhat[:, 0]
    #         # invert scaling for actual
    #         test_y = test_y.reshape((len(test_y), 1))
    #         inv_y = concatenate((test_y, test_X[:, -29:]), axis=1)
    #         inv_y = scaler.inverse_transform(inv_y)
    #         inv_y = inv_y[:, 0]
    #         # calculate RMSE
    #         mape = mean_absolute_percentage_error(inv_y, inv_yhat)
    #         print("Thresh=%.3f, n=%d, MAPE: %.2f%%" % (thresh, select_X_train.shape[1], mape))
