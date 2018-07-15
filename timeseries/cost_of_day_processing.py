#-*-coding:utf-8-*-
'''
加入星期特征，以及月份特征，季度特征，是否工作日，是否节假日
'''
from random import randint
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from datetime import datetime
from chinese_calendar import is_workday, is_holiday
import chinese_calendar as calendar    # 也可以这样 import

#####生成所有的日期数据
def getBetweenDate(start,end):
    datestart = datetime.strptime(start, '%Y-%m-%d')
    dateend = datetime.strptime(end, '%Y-%m-%d')
    date=[]
    while datestart < dateend:
        datestart += timedelta(days=1)
        date.append(datestart.strftime('%Y-%m-%d'))
    return date

#####获得该日期所属季度
def getSeason(month):
    if month in range(1,4):
        return 1
    elif month in range(4,7):
        return 2
    elif month in range(7,10):
        return 3
    else:
        return 4

####对缺失数据进行插值填充(以其前3天和后3天的均值进行填充)
def getMissingData(new_data,cur_date):
    # print("当前缺失日期：%s" % (cur_date))
    ##前3天
    date = datetime.strptime(cur_date, '%Y-%m-%d')
    sum_fees=0
    h_count=0
    count=0
    for i in range(-3,0):
        datestart = date+timedelta(days=i)
        datestart=datestart.strftime('%Y-%m-%d')
        if datestart not in new_data['date']:
            sum_fees=sum_fees+0
        else:
            count=count+1
            sum_fees=sum_fees+new_data.ix[datestart, 'group_fees']
            h_count=h_count+new_data.ix[datestart, 'h_count']
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
            h_count = h_count + new_data.ix[datestart, 'h_count']
        # print("后%d天:%s" % (i, datestart))
    if count==0:
       ####用该月已有数据的均值代替
       avg_fees=np.mean(new_data[cur_date[0:7]].group_fees)
       h_count=np.mean(new_data[cur_date[0:7]].h_count)
    else:
        avg_fees=sum_fees/count
        avg_count=h_count/count
    return avg_fees,avg_count

data_input= pd.read_csv('dataset/workerCostByDay.csv', header=None)
data_input.columns=['date','total_fees','group_fees','hospital_fees','h_groupfees','menzhen_fees','m_groupfees','h_count','m_count','avg_hgroupfees','avg_mgroupfees']
data=data_input[['date','h_count','h_groupfees','m_count','m_groupfees','group_fees']]

new_data= pd.DataFrame(columns=('date', 'weekday', 'month','season','weekday_or_not','holiday_or_not','h_count','h_groupfees','m_count','m_groupfees','group_fees'))
i=0
new_data.index=new_data['date'].tolist()
for index, row in data.iterrows():
    date=str(int(row['date']))
    date=date[0:4]+'-'+date[4:6]+'-'+date[6:8]
    new_date = datetime.strptime(date, "%Y-%m-%d")
    weekday=new_date.weekday()   ###判断是星期几
    month=new_date.month         ###获得对应的月份
    season = getSeason(month)  ####对应的季度
    ####判断是否是工作日
    if (is_workday(new_date)):
        weekday_or_not = 1
    else:
        weekday_or_not = 0
    ####判断是否是节假日
    if (is_holiday(new_date)):
        holiday_or_not = 1
    else:
        holiday_or_not = 0
    line=[new_date,int(weekday),month,season,weekday_or_not,holiday_or_not,row['h_count'],row['h_groupfees'],row['m_count'],row['m_groupfees'],row['group_fees']]
    new_data.loc[i]=line
    i=i+1

new_data.to_csv('dataset/new_cost_of_month_weekday.csv',index=False)



#####重新设置索引
# new_data.index=new_data['date'].tolist()
# ####获取日期集合
# date_set=getBetweenDate("2005-12-31","2016-06-29")
# imputation_data=pd.DataFrame(columns=('date', 'weekday', 'month','season','weekday_or_not','holiday_or_not','h_count','m_count','group_fees'))
# error_count=0
# i=0
# for date in date_set:
#     new_date = datetime.strptime(date, "%Y-%m-%d")
#     day = new_date.weekday()  ###判断是星期几
#     month = new_date.month  ###获得对应的月份
#     season=getSeason(month)   ####对应的季度
#     ####判断是否是工作日
#     if(is_workday(new_date)):
#         weekday_or_not=1
#     else:
#         weekday_or_not=0
#     ####判断是否是节假日
#     if(is_holiday(new_date)):
#         holiday_or_not=1
#     else:
#         holiday_or_not=0
#     # if date in new_data['date']:
#     h_count=new_data.ix[new_date,'h_count']
#     m_count = new_data.ix[new_date, 'm_count']
#     group_fees=new_data.ix[new_date,'group_fees']
#     line = [new_date, int(day), month,season,weekday_or_not,holiday_or_not,h_count,m_count,group_fees]
#     # else:
#     #     print(new_date)
#     #     error_count=error_count+1
#     #     group_fees,h_count,m_count=getMissingData(new_data,date)
#     #     line = [new_date, int(day), month,season,weekday_or_not,holiday_or_not,h_count,m_count,group_fees]
#     imputation_data.loc[i] = line
#     i=i+1

# print("缺失天数：%d"%(error_count))
# imputation_data.to_csv('dataset/new_cost_of_month_weekday.csv',index=False)
