#coding:utf-8
'''
加入趋势的信息
'''
from datetime import datetime, timedelta, time

import pandas as pd
data_input= pd.read_csv('dataset/new_cost_of_month_weekday.csv')
data_input.columns=['date', 'weekday', 'month','season','weekday_or_not','holiday_or_not','h_count','h_groupfees','m_count','m_groupfees','group_fees']
new_data= pd.DataFrame(columns=('date', 'weekday', 'month','season','weekday_or_not','holiday_or_not','trend','h_count','h_groupfees','m_count','m_groupfees','group_fees'))
data_input.index=data_input['date'].tolist()
flag=False
i=0
for index, row in data_input.iterrows():
    date=row['date']
    now_date=datetime.strptime(date, "%Y-%m-%d")
    last_date=now_date-timedelta(days=1)
    now_date=datetime.strftime(now_date,"%Y-%m-%d")
    last_date=datetime.strftime(last_date,"%Y-%m-%d")
    if flag==False:
        trend=1   ####上升的趋势
        flag=True
    else:
        last_groupfees = data_input.ix[last_date, 'group_fees']
        now_groupfees = data_input.ix[now_date, 'group_fees']
        if now_groupfees>last_groupfees:
            trend=1 ###上升的趋势
        else:
            trend=-1 ###下降的趋势
        # trend=(now_groupfees-last_groupfees)/last_groupfees
    line = [row['date'],row['weekday'], row['month'], row['season'], row['weekday_or_not'], row['holiday_or_not'],trend, row['h_count'], row['h_groupfees'],
            row['m_count'], row['m_groupfees'], row['group_fees']]
    new_data.loc[i] = line
    i = i + 1

new_data.to_csv('dataset/new_cost_of_month_trend.csv',index=False)