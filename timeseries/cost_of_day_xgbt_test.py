# plot decision tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from timeseries.time_series_processing import *

# load dataset
####读入介入星期特征以及月份特征的数据
if __name__=='__main__':
    ####生成数据集
    n_in=2
    n_history=1
    n_out=1
    n_train_sample=-90
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
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_in, n_out)
    print(reframed.shape)
    # split into train and test sets
    new_values = reframed.values
    train = new_values[:n_train_sample, :]
    test = new_values[n_train_sample:-30, :]
    # split into input and outputs
    train_X, train_y = train[:, :-5], train[:, -1]
    test_X, test_y = test[:, :-5], test[:, -1]
    print(train_X.shape, len(train_X), train_y.shape)

    ####进行数据建模
    xgbt=XGBRegressor()
    parameters = {'n_estimators': [80,90,100,110,120], 'learning_rate':[0.01,0.05,0.1,0.2],'max_depth':[3,4,5]}
    grid_search = GridSearchCV(estimator=xgbt, param_grid=parameters, cv=5)
    grid_search.fit(train_X, train_y)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    ####对模型进行打分
    preds_train =grid_search.predict(train_X)
    print("模型打分情况：",metrics.r2_score(train_y,preds_train))

    preds_test = grid_search.predict(test_X)
    # 对预测结果进行展示
    # plot_mul_results(scaler,test_X,yhat,test_y)
    reframed = series_to_supervised(scaled, n_in, n_out)
    print(reframed)
    # split into train and test sets
    new_values = reframed.values

    train = new_values[:n_train_sample, :]
    test = new_values[n_train_sample:-30, :]
    # split into input and outputs
    train_X, train_y = train[:, :-5], train[:, -1]
    test_X, test_y = test[:, :-5], test[:, -1]
    print(train_X.shape, len(train_X), train_y.shape)
    xgbt = XGBRegressor()
    parameters = {'n_estimators': [80, 90, 100, 110, 120], 'learning_rate': [0.01, 0.05, 0.1, 0.2],
                  'max_depth': [3, 4, 5]}
    grid_search = GridSearchCV(estimator=xgbt, param_grid=parameters, cv=5)
    grid_search.fit(train_X, train_y)
    yhat = grid_search.predict(test_X)
    # 对预测结果进行展示
    plot_mul_results(scaler, test_X, yhat, test_y)