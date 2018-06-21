#-*-coding:utf-8-*-
'''
native_test
'''
import time
from baseline.Naive_Lof import outliers
from baseline.Naive_Lof import LOF
from fraud_detection import data_preparation

######数据准备工作
data=data_preparation.generate_data()
# start = time.time()
lof = outliers(5, data)
for outlier in lof:
    print(outlier["lof"],outlier["instance"])
# lof = LOF(data)
# for instance in data:
#     value = lof.local_outlier_factor(5, instance)
#     print(value, instance)
# print ("---------------------")
# print ("time cost: %s seconds." % (time.time() - start))
# print ("---------------------")