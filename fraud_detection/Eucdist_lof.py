import time

from fraud_detection import data_preparation
from baseline.Euc_Dist_Optimized_Lof import lof


######数据准备工作
data=data_preparation.generate_data()
# lof = outliers(5, data)
# for outlier in lof:
#     print(outlier["lof"],outlier["instance"])
start = time.time()
k=5
predicted_outliers = lof(data, 5, outlier_threshold = 2)
print(predicted_outliers)
print ("---------------------")
print ("time cost: %s seconds." % (time.time() - start))
print ("---------------------")