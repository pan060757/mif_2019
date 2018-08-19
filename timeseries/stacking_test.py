from mlxtend.regressor import LinearRegression
from mlxtend.regressor import StackingRegressor
from sklearn import metrics, model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from timeseries.time_series_processing import *

if __name__=='__main__':
    ####生成数据集
    scaler,train_X,train_y,test_X,test_y=generate_mul_dataset(2,1,1,-90)
    xgbt = XGBRegressor()

    gbdt = GradientBoostingRegressor()

    ####加入随机森林模型
    rf = RandomForestRegressor()

    ####加入线性模型
    lr = LinearRegression()

    sclf = StackingRegressor(regressors=[rf,gbdt,xgbt],
                             meta_regressor=rf)
    # 对预测结果进行展示
    for clf, label in zip([rf, gbdt,sclf],
                          ['model1',
                           'Random Forest',
                           'StackingClassifier']):
        clf.fit(train_X,train_y)
        lr_train = clf.predict(train_X)
        ####对模型进行打分
        print("模型打分情况：", metrics.r2_score(train_y, lr_train))
        yhat = clf.predict(test_X)
        # 对预测结果进行展示
        plot_mul_results(scaler, test_X, yhat, test_y)
