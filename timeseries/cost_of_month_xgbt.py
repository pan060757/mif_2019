# plot decision tree
from matplotlib import pyplot
from numpy import loadtxt, sort, concatenate
from pandas import concat
from pandas import read_csv
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
    # load dataset
    # 加载数据
    data_input = read_csv('dataset/workerCostByMonth.csv', header=None)
    #####
    data_input.columns = ['date', 'total_fees', 'group_fees', 'hospital_fees', 'h_groupfees', 'menzhen_fees',
                          'm_groupfees', 'hospital_count', 'menzhen_count', 'avg_hgroupfees', 'avg_mgroupfees']
    #####只提取一个特征
    values = data_input['group_fees']
    values = values.reshape(-1, 1)
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

    eval_set=[(test_X,test_y)]
    # fit model no training data
    model = XGBRegressor(learning_rate=0.1,n_estimators=100,max_depth=4)
    model.fit(train_X, train_y,eval_metric=['rmse'],eval_set=eval_set,verbose=True)
    # feature importance
    print(model.feature_importances_)
    # plot feature importance
    # plot_importance(model)
    # pyplot.show()

    # n_estimators数目
    # n_estimators = range(50, 400, 50)
    # param_grid = dict(n_estimators=n_estimators)
    # grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1)
    # grid_result = grid_search.fit(train_X, train_y)
    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    # # plot
    # pyplot.errorbar(n_estimators, means, yerr=stds)
    # pyplot.title("XGBoost n_estimators vs Log Loss")
    # pyplot.xlabel('n_estimators')
    # pyplot.ylabel('mean_absolute_error')
    # pyplot.savefig('dataset/gbdt-n_estimators.png')
    #
    #
    # ####最大深度(3)
    # max_depth = range(1, 11, 2)
    # print(max_depth)
    # param_grid = dict(max_depth=max_depth)
    # # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    # grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    # grid_result = grid_search.fit(train_X, train_y)
    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    # # plot
    # pyplot.errorbar(max_depth, means, yerr=stds)
    # pyplot.title("XGBoost max_depth vs Log Loss")
    # pyplot.xlabel('max_depth')
    # pyplot.ylabel('Log Loss')
    # pyplot.savefig('dataset/xgboost-max_depth.png')


    # ##### n_estimators和max_depth
    # n_estimators = [50, 100, 150, 200]
    # max_depth = [2, 4, 6, 8]
    # print(max_depth)
    # param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
    # # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    # grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    # grid_result = grid_search.fit(train_X, train_y)
    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    # # plot results
    # scores = np.array(means).reshape(len(max_depth), len(n_estimators))
    # for i, value in enumerate(max_depth):
    #     pyplot.plot(n_estimators, scores[i], label='depth: ' + str(value))
    # pyplot.legend()
    # pyplot.xlabel('n_estimators')
    # pyplot.ylabel('Log Loss')
    # pyplot.savefig('dataset/n_estimators_vs_max_depth.png')

    ###learning rate
    # learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    # param_grid = dict(learning_rate=learning_rate)
    # # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    # grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    # grid_result = grid_search.fit(train_X, train_y)
    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    # # plot
    # pyplot.errorbar(learning_rate, means, yerr=stds)
    # pyplot.title("XGBoost learning_rate vs Log Loss")
    # pyplot.xlabel('learning_rate')
    # pyplot.ylabel('Log Loss')
    # pyplot.savefig('dataset/learning_rate.png')

    #####调节learning_rate和n_estimators
    # n_estimators = [100, 200, 300, 400, 500]
    # learning_rate = [0.0001, 0.001, 0.01, 0.1]
    # param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    # # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    # grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    # grid_result = grid_search.fit(train_X, train_y)
    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    # # plot results
    # scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
    # for i, value in enumerate(learning_rate):
    #     pyplot.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
    # pyplot.legend()
    # pyplot.xlabel('n_estimators')
    # pyplot.ylabel('Log Loss')
    # pyplot.savefig('dataset/n_estimators_vs_learning_rate.png')

    yhat = model.predict(test_X)
    # invert scaling for forecast
    yhat = yhat.reshape(len(yhat), 1)
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
    #         inv_yhat = concatenate((yhat, test_X[:, -27:]), axis=1)
    #         inv_yhat = scaler.inverse_transform(inv_yhat)
    #         inv_yhat = inv_yhat[:, 0]
    #         # invert scaling for actual
    #         test_y = test_y.reshape((len(test_y), 1))
    #         inv_y = concatenate((test_y, test_X[:, -27:]), axis=1)
    #         inv_y = scaler.inverse_transform(inv_y)
    #         inv_y = inv_y[:, 0]
    #         # calculate RMSE
    #         mape = mean_absolute_percentage_error(inv_y, inv_yhat)
    #         print("Thresh=%.3f, n=%d, MAPE: %.2f%%" % (thresh, select_X_train.shape[1], mape))
