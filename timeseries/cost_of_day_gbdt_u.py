from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

####读入介入星期特征以及月份特征的数据
from timeseries.time_series_processing import *

if __name__=='__main__':
    ####生成数据集
    scaler,train_X,train_y,test_X,test_y=generate_uni_dataset(2,1,1,-90)
    param_test2 = {'max_depth':range(3,14,2),'min_samples_split':range(100,801,200)}
    model= GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=120,learning_rate=0.05, min_samples_split=250, min_samples_leaf=20,max_depth=9,max_features='sqrt', subsample=0.8,random_state=10), param_grid = param_test2,iid=False,cv=5)
    model.fit(train_X,train_y)
    print(model.grid_scores_)
    print(model.best_params_)
    ####对模型进行打分
    preds_train = model.predict(train_X)
    print("模型打分情况：", metrics.r2_score(train_y, preds_train))

    # invert scaling for forecast
    yhat = model.predict(test_X)
    # 对预测结果进行展示
    plot_uni_results(scaler, yhat, test_y)

