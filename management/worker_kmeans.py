#-*-coding:utf-8-*-
'''
对职工数据进行聚类操作
'''
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('dataset/worker_processing.csv',header=None)
#######首先进行数据预处理操作
# print(timeseries.count())          ####是否存在缺失数据

###列重命名操作
data.columns=['workerNum','year','age','state','workplace','gender','wage','chroric','days','times','single_fees']

####提取需要的列
#指定年份
data_year=data[data['year']==2015]
#丢掉不需要的列(去掉个人编号、年份)
data_select=data_year.drop(['workerNum','year'],1)

##对个别列进行哑变量处理
state_dummy=pd.get_dummies(data_select.state,prefix='state')
data_select=data_select.join(state_dummy)

gender_dummy=pd.get_dummies(data_select.gender,prefix='gender')
data_select=data_select.join(gender_dummy)

workplace_dummy=pd.get_dummies(data_select.workplace,prefix='workplace')
data_select=data_select.join(workplace_dummy)

chroric_dummy=pd.get_dummies(data_select.chroric,prefix='chroric')
data_select=data_select.join(chroric_dummy)

####去除多余的列
data_select=data_select.drop(['state','gender','workplace','chroric'],1)
print(data_select.head())

#对个别列进行数据标准化操作
print('得到标准化之后的数据：')
min_max_scaler = preprocessing.MinMaxScaler()
std_data=min_max_scaler.fit_transform(data_select)
print(std_data)


#####完成聚类操作
####用来评估簇的个数是否合适，距离越小说明簇分的越好
for i in range(1,30,1):
    model = KMeans(n_clusters = i,init = 'k-means++')
    s=model.fit(std_data)
    print("%d,%.2f"%(i,model.inertia_))

N=5
####找出最佳的K值
model = KMeans(n_clusters = 5,init = 'k-means++')
model.fit(std_data)
####显示聚类中心
print(model.cluster_centers_)
#####进行预测
print(model.predict(std_data))