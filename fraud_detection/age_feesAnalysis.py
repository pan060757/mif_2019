#-*-coding:utf-8
'''
基于年龄分布拟合的残差异常识别
'''
import pandas as pd
import matplotlib.pyplot as plt

def computeLOF(lof_list,groupfees,age_groupfees):
    for index,row in groupfees.iterrows():
        age=row['age']
        groupfees=row['group_fees']
        mean_groupfees=age_groupfees[age]
        lof=(groupfees-mean_groupfees)/mean_groupfees
        lof_list[index]=lof
    return lof_list

data_input=pd.read_csv("dataset/dataOfYZJPTC.csv")
data_input.columns=['worker_No','identity','worker_place','age','sex','wage','hospital_No','hospital_grade','days','drug_fees','line','ratio','chroric_or_not','in_hospital','out_hospital','trement_fees','bed_fees','operation_fees'
    ,'care_fees','material_fees','group_fees']
groupfees=data_input[['age','group_fees']]
age_groupfees=data_input.groupby('age')['group_fees'].mean()
# print(age_groupfees)

# plt.plot(age_groupfees.index,age_groupfees,marker='s')
# plt.xlabel(u'年龄')
# plt.ylabel(u'平均住院费用')
# plt.title(u'平均住院费用的年龄分布情况')
# plt.grid(True)
# plt.show()

####计算异常程度
lof_list={}
lof_list=computeLOF(lof_list,groupfees,age_groupfees)
for key,value in lof_list.items():
    print(key,value)
