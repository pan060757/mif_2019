#coding:utf-8
'''
一单大额费用分布情况(异常分数计算)
'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
def computedLof(lof_list,group_fees,median_fees):
    for key,value in group_fees.items():
        lof=(value-median_fees)/median_fees
        lof_list[key]=lof
    return lof_list

data_input=pd.read_csv("dataset/dataOfYZJPTC.csv")
data_input.columns=['worker_No','identity','worker_place','age','sex','wage','hospital_No','hospital_grade','days','drug_fees','line','ratio','chroric_or_not','in_hospital','out_hospital','trement_fees','bed_fees','operation_fees'
    ,'care_fees','material_fees','group_fees']
group_fees=data_input['group_fees']

# plt.hist(group_fees,100)
# plt.xlabel(u'住院费用')
# plt.ylabel(u'频次')
# plt.title(u'住院费用支出频数分布图')
# plt.grid(True)
# plt.show()

####求一组数据的平均值#####
mean_fees=np.mean(group_fees)
print(mean_fees)
####求一组数据的中位数#####
median_fees=np.median(group_fees)
print(median_fees)

####异常程度计算######
lof_list={}
lof_list=computedLof(lof_list,group_fees,median_fees)
for key,value in lof_list.items():
    print(key,value)