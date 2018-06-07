#-*-coding:utf-8 -*-
'''
住院记录分析
'''
import pandas as pd

########错误记录情况统计
count=0        ####记录列数超过30的错误数据条数
# with open("dataset/worker_hospital.csv",'r',encoding='utf8') as f:
#     for line in f:
#         lines=line.split(",")
#         if(len(lines)>28):
#             count=count+1
#     print("错误条数:%d"%(count))   ###总共错误条数：8条

####统计整个数据集的数据缺失情况
data_input=pd.read_csv("dataset/worker_hospital.csv",header=None)
data_input.columns=['region_num','worker_num','hos_num','hospital_num',
                    'hospital_type','hospital_grade','total_fees','jl_fees',
                    'yl_fees','fjb_fees','drug_fees','jlyp_fees','ylyp_fees',
                    'fjbyp_fees','line','ratio','single_fees','group_fees','bbzf_fees',
                    'gwybz_fees','grzf_fees','in_hospital','out_hospital','pay_type',
                    'in_diseaseNum','in_diseaseName','in_diseaseNum','in_diseaseName']
print(data_input.count())