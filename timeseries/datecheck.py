#-*-coding:utf-8 -*-
'''
检查获取到的住院日期是否完整
'''

#-*-coding:utf-8 -*-
from operator import add

'''
统计每月的住院总费用，统筹费用支出，门诊费用，门诊统筹费用支出
'''
import datetime

import re
from pyspark import SparkContext
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def hospitalProcessing(line):
    line = line.encode('utf-8').split(',')
    for i in range(6, 20):
        if (line[i] == ""):
            line[i] = '0'
    if (line[21] != ""):  ###可能存在未记录出院时间和住院时间的住院记录
        inHospital = line[21]
        s = inHospital.strip("").split('-')
        s[1] = re.sub("\D", "", s[1])  ##提取其中数字部分
        if len(s[0]) < 2:
            s[0] = '0' + s[0]
        if len(s[1]) < 2:
            s[1] = '0' + s[1]
        day='20'+s[2]+s[1]+s[0]
        return (day, (float(line[6]), float(line[17]), 1))
    else:
        return (str(999999),1)

sc=SparkContext()
# hospital=sc.textFile('/mif/data_new/worker_hospital.txt')
# ####（（日期，(总费用，统筹费用支出,住院次数次数)）
# hospital=hospital.map(hospitalProcessing)\
#     .filter(lambda (key,value):(isinstance(value,int)==False and key>'2006')) \
#     .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])) \
#     .sortByKey()

men_zhen=sc.textFile('/mif/data_new/worker_menzhen.txt')
####（（日期，(总费用，统筹费用支出，门诊人次)）
men_zhen=men_zhen.map(lambda line:line.encode('utf-8').split(','))\
    .filter(lambda line:line[5]!="" and line[12]!="")\
    .map(lambda line:(line[15],(float(line[5]),float(line[12]),1)))\
    .reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1],a[2]+b[2]))\
    .sortByKey()


###（日期)
#######通过检查可知，住院数据比较完整，门诊数据不完整
out=open('output/m_dateCheck.csv','w+')
for (key,value)in men_zhen.collect():
    out.write("%s\n"%key)
out.close()

