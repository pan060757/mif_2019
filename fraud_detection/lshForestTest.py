######数据准备工作
import time

from fraud_detection import data_preparation
from fraud_detection.baseline.lshForest_Lof import lof

data=data_preparation.generate_data()
start = time.time()
k=5
predicted_outliers = lof(data,k, outlier_threshold =1.5)
print(predicted_outliers)
print("捕获的异常点数：",len(predicted_outliers))
print ("---------------------")
print ("time cost: %s seconds." % (time.time() - start))
print ("---------------------")
