#-*-coding:utf-8-*-
'''
加入星期特征，以及月份特征，
'''
from random import randint
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

#####生成所有的日期数据
def getBetweenDate(start,end):
    datestart = datetime.strptime(start, '%Y-%m-%d')
    dateend = datetime.strptime(end, '%Y-%m-%d')
    date=[]
    while datestart < dateend:
        datestart += timedelta(days=1)
        date.append(datestart.strftime('%Y-%m-%d'))
    return date

####对缺失数据进行插值填充(以其前3天和后3天的均值进行填充)
def getMissingData(new_data,cur_date):
    # print("当前缺失日期：%s" % (cur_date))
    ##前3天
    date = datetime.strptime(cur_date, '%Y-%m-%d')
    sum_fees=0
    count=0
    for i in range(-3,0):
        datestart = date+timedelta(days=i)
        datestart=datestart.strftime('%Y-%m-%d')
        if datestart not in new_data['date']:
            sum_fees=sum_fees+0
        else:
            count=count+1
            sum_fees=sum_fees+new_data.ix[datestart, 'group_fees']
        # print("前%d天:%s"%(abs(i),datestart ))
    ##后3天
    for i in range(1, 4):
        datestart = date + timedelta(days=i)
        datestart = datestart.strftime('%Y-%m-%d')
        if datestart not in new_data['date']:
            sum_fees = sum_fees + 0
        else:
            count=count+1
            sum_fees = sum_fees + new_data.ix[datestart, 'group_fees']
        # print("后%d天:%s" % (i, datestart))
    if count==0:
       ####用该月已有数据的均值代替
       avg_fees=np.mean(new_data[cur_date[0:7]].group_fees)
    else:
        avg_fees=sum_fees/count
    return avg_fees

data_input= pd.read_csv('dataset/workerCostByDay.csv', header=None)
data_input.columns=['date','total_fees','group_fees','hospital_fees','h_groupfees','menzhen_fees','m_groupfees','hospital_count','menzhen_count','avg_hgroupfees','avg_mgroupfees']
data=data_input[['date','group_fees']]

new_data= pd.DataFrame(columns=('date', 'weekday', 'month','group_fees'))
i=0
for index, row in data.iterrows():
    date=str(int(row['date']))
    date=date[0:4]+'-'+date[4:6]+'-'+date[6:8]
    new_date = datetime.strptime(date, "%Y-%m-%d")
    weekday=new_date.weekday()   ###判断是星期几
    month=new_date.month         ###获得对应的月份
    line=[new_date,int(weekday),month,row['group_fees']]
    new_data.loc[i]=line
    i=i+1

#####重新设置索引
new_data.index=new_data['date'].tolist()
####获取日期集合
date_set=getBetweenDate("2006-01-03","2016-07-01")
imputation_data=pd.DataFrame(columns=('date', 'weekday', 'month','group_fees'))
error_count=0
i=0
for date in date_set:
    new_date = datetime.strptime(date, "%Y-%m-%d")
    weekday = new_date.weekday()  ###判断是星期几
    month = new_date.month  ###获得对应的月份
    if date in new_data['date']:
        line = [new_date, int(weekday), month, new_data.ix[new_date,'group_fees']]
    else:
        error_count=error_count+1
        group_fees=getMissingData(new_data,date)
        line = [new_date, int(weekday), month,group_fees]
    imputation_data.loc[i] = line
    i=i+1

print("缺失天数：%d"%(error_count))
imputation_data.to_csv('dataset/new_cost_of_month_weekday.csv',index=False)
###进行缺失值填充





# new_data.to_csv('dataset/cost_of_month_weekday.csv',index=False)

import json
import requests
###判断是否节假日
# date = "20180504"
# server_url = "http://www.easybots.cn/api/holiday.php?d="
#
# vop_response = requests.get(server_url+date)
# print(vop_response.text)
# if vop_data[date] == '0':
#     print("this day is weekday")
# elif vop_data[date] == '1':
#     print('This day is weekend')
# elif vop_data[date] == '2':
#     print('This day is holiday')
# else:
#     print('Error')