# plot decision tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from timeseries.time_series_processing import *

# load dataset
####读入介入星期特征以及月份特征的数据
if __name__=='__main__':
    ####生成数据集
    scaler,train_X,train_y,test_X,test_y=generate_uni_dataset(2,1,1,-90)
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

    yhat = grid_search.predict(test_X)
    # 对预测结果进行展示
    plot_uni_results(scaler,test_X,yhat,test_y)