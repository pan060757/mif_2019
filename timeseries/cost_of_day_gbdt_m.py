from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

####读入介入星期特征以及月份特征的数据
from timeseries.time_series_processing import *

if __name__=='__main__':
    ####生成数据集
    scaler,train_X,train_y,test_X,test_y=generate_mul_dataset(2,1,1,-90)
    param_test2 = {'max_depth':range(3,14,2),'min_samples_split':range(100,801,200)}
    search= GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=120,learning_rate=0.05, min_samples_leaf=20,max_features='sqrt', subsample=0.8,random_state=10), param_grid = param_test2,iid=False,cv=5)
    search.fit(train_X,train_y)
    ####进行特征选择
    print(search.grid_scores_)
    print(search.best_params_)
    model=GradientBoostingRegressor(n_estimators=120,learning_rate=0.05, min_samples_split=100, min_samples_leaf=20,max_features='sqrt', subsample=0.8,random_state=10,max_depth=9)
    # model=model.fit(train_X, train_y)
    ####进行特征选择
    #####树模型进行特征降维
    # train_X,test_X=selectFeatures(train_X,test_X,model)
    # ###normalize features
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled = scaler.fit_transform(train_X)
    # model = model.fit(train_X, train_y)

    #####利用PCA进行特征降维
    est=PCA(n_components=45)  ####11.647
    train_X=est.fit_transform(train_X)
    test_X=est.transform(test_X)
    model = model.fit(train_X, train_y)
    preds_train = model.predict(train_X)
    ####将结果写入文件
    out = open("dataset/training_set.csv", "w+")
    for i in range(len(preds_train)):
        out.write(str(preds_train[i])+'\n')
    out.close()
    ####对模型进行打分
    print("模型打分情况：", metrics.r2_score(train_y, preds_train))

    # invert scaling for forecast
    yhat = model.predict(test_X)
    # 对预测结果进行展示
    plot_mul_results(scaler, test_X, yhat, test_y)

