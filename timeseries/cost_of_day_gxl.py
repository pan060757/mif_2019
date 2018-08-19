#coding:utf-8
'''
进行模型的融合（gbdt+xgbt+LinearRegression）
'''
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

####读入介入星期特征以及月份特征的数据
from sklearn.svm import SVR
from xgboost import XGBRegressor

from timeseries.time_series_processing import *

if __name__=='__main__':
    ####生成数据集
    scaler,train_X,train_y,test_X,test_y=generate_mul_dataset(2,1,1,-90)
    #####xgbt模型
    ####进行数据建模
    grid_search = XGBRegressor()
    grid_search.fit(train_X,train_y)
    # xgbt = XGBRegressor(gamma=0.1)
    # parameters = {'n_estimators': [80, 90, 100, 110, 120], 'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #               'max_depth': range(3,10,2),'reg_alpha':[1e-5,1e-2,0.1,1]}
    # grid_search = GridSearchCV(estimator=xgbt, param_grid=parameters, cv=5)
    # grid_search.fit(train_X, train_y)

    ####对模型进行打分
    xgbt_train = grid_search.predict(train_X)
    print("xgbt-模型打分情况：", metrics.r2_score(train_y, xgbt_train))
    xgbt_test = grid_search.predict(test_X)
    plot_mul_results(scaler, test_X, xgbt_test, test_y)

    #####gbdt模型
    # param_test2 = {'max_depth':range(3,14,2),'min_samples_split':range(100,801,200)}
    # search= GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=120,learning_rate=0.05, min_samples_leaf=20,max_features='sqrt', subsample=0.8,random_state=10), param_grid = param_test2,iid=False,cv=5)
    # search.fit(train_X,train_y)
    model=GradientBoostingRegressor(n_estimators=120,learning_rate=0.05, min_samples_split=100, min_samples_leaf=20,max_features='sqrt', subsample=0.8,random_state=10,max_depth=9)
    #####利用PCA进行特征降维
    est=PCA(n_components=45)  ####11.647
    train_X=est.fit_transform(train_X)
    test_X=est.transform(test_X)
    model = model.fit(train_X, train_y)
    gbdt_train = model.predict(train_X)
    ####对模型进行打分
    print("gbdt-模型打分情况：", metrics.r2_score(train_y, gbdt_train))
    # invert scaling for forecast
    gbdt_test = model.predict(test_X)
    # 对预测结果进行展示
    plot_mul_results(scaler, test_X, gbdt_test, test_y)

    ####加入随机森林模型
    model = RandomForestRegressor()
    model.fit(train_X, train_y)
    rf_train = model.predict(train_X)
    ####对模型进行打分
    print("randomforest-模型打分情况：", metrics.r2_score(train_y, rf_train))
    # invert scaling for forecast
    rf_test = model.predict(test_X)
    # 对预测结果进行展示
    plot_mul_results(scaler, test_X, rf_test, test_y)

    ####加入线性模型
    model = LinearRegression()
    model.fit(train_X, train_y)
    lr_train = model.predict(train_X)
    ####对模型进行打分
    print("LR线性模型打分情况：", metrics.r2_score(train_y, lr_train))
    # invert scaling for forecast
    lr_test = model.predict(test_X)
    # 对预测结果进行展示
    plot_mul_results(scaler, test_X, lr_test, test_y)

    #####利用前两次建模的结果作为LR线性模型此次输入
    #####
    new_train=pd.DataFrame(columns=['xgbt_value','rf_value'])
    new_train['xgbt_value']=pd.Series(xgbt_train)
    # new_train['gbdt_value'] = pd.Series(gbdt_train)
    # new_train['lr_value'] = pd.Series(lr_train)
    new_train['rf_value'] = pd.Series(rf_train)
    new_train_x = pd.concat([pd.DataFrame(train_X),new_train], axis=1)
    #####再次进行xgbt模型
    model = XGBRegressor(gamma=0.1)
    # model=LinearRegression()
    model.fit(new_train_x, train_y)
    ####对模型进行打分
    mm_train = model.predict(new_train_x)
    print("stacking模型打分情况：", metrics.r2_score(train_y, mm_train))
    new_test = pd.DataFrame(columns=['xgbt_value', 'rf_value'])
    new_test['xgbt_value'] = pd.Series(xgbt_test)
    # new_test['gbdt_value'] = pd.Series(gbdt_test)
    # new_test['lr_value'] = pd.Series(lr_test)
    new_test['rf_value'] = pd.Series(rf_test)
    new_test_x = pd.concat([pd.DataFrame(test_X), new_test], axis=1)
    yhat = model.predict(new_test_x)
    plot_mul_results(scaler, test_X, yhat, test_y)