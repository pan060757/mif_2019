from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV

####读入介入星期特征以及月份特征的数据
from sklearn.neural_network import MLPRegressor

from timeseries.time_series_processing import *




if __name__=='__main__':
    ####生成数据集
    scaler,train_X,train_y,test_X,test_y=generate_mul_dataset(2,1,1,-90)
    model=LinearRegression()
    model.fit(train_X,train_y)
    ####对模型进行打分
    preds_train = model.predict(train_X)
    print("模型打分情况：", metrics.r2_score(train_y, preds_train))

    # invert scaling for forecast
    yhat = model.predict(test_X)
    # 对预测结果进行展示
    plot_mul_results(scaler, test_X, yhat, test_y)