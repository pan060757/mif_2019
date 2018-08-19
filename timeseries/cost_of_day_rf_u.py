from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from timeseries.time_series_processing import *

if __name__=='__main__':
    ####生成数据集
    scaler,train_X,train_y,test_X,test_y=generate_uni_dataset(2,1,1,-90)
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
    # 对预测结果进行展示
    plot_uni_results(scaler,yhat,test_y)

