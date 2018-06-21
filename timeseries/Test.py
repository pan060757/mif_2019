from datetime import datetime


# ####判断今天星期几
# date='2018-06-03'
# new_date = datetime.strptime(date, "%Y-%m-%d")
# weekday = new_date.weekday()  ###判断是星期几
# print(weekday)


import json
# import requests
# ###判断是否节假日(不太准)
# date = "20161010"
# server_url = "http://www.easybots.cn/api/holiday.php?d="
#
# vop_data = requests.get(server_url+date)
# print(vop_data.content)

# if vop_data[date] == '0':
#     print("this day is weekday")
# elif vop_data[date] == '1':
#     print('This day is weekend')
# elif vop_data[date] == '2':
#     print('This day is holiday')
# else:
#     print('Error')


####
# 判断 2018年4月30号 是不是节假日
from datetime import datetime
from chinese_calendar import is_workday, is_holiday
import chinese_calendar as calendar    # 也可以这样 import
# april_last = datetime.date(2016, 4,5)
# date='2015-05-17'
# april_last = datetime.strptime(date, "%Y-%m-%d")
# print(is_workday(april_last))
# print(is_holiday(april_last))
# holiday_name = calendar.get_holiday_detail(april_last)
# print(calendar.Holiday.labour_day.value, holiday_name)

# 或者在判断的同时，获取节日名
# import numpy as np
# list1=[np.array([1,2,3]),np.array([2,3,4]),np.array([4,5,6])]
# list1.remove(list1[0])
# print(list1)

####
