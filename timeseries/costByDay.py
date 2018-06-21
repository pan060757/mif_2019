#-*-coding:utf-8 -*-
'''
统计每天的住院总费用，统筹费用支出，门诊费用，门诊统筹费用支出
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
        return(day,(float(line[6]),float(line[17]),1))
    else:
        return (str(999999),1)

###（日期,(住院总费用，住院统筹支付，门诊总费用，门诊统筹支出，住院人次，门诊人次))
def hospital_menzhen((key,value)):
    if(value[1]==None):
        return(key,(value[0][0],value[0][1],0,0,value[0][2],0))
    else:
        return(key,(value[0][0],value[0][1],value[1][0],value[1][1],value[0][2],value[1][2]))

sc=SparkContext()
hospital=sc.textFile('/mif/data_new/worker_hospital.txt')
####（（日期，(总费用，统筹费用支出,住院次数次数)）
hospital=hospital.map(hospitalProcessing)\
    .filter(lambda (key,value):(isinstance(value,int)==False and key>'2006'))\
    .reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1],a[2]+b[2]))\
    .sortByKey()

men_zhen=sc.textFile('/mif/data_new/worker_menzhen.txt')
####（（日期，(总费用，统筹费用支出，门诊人次)）
men_zhen=men_zhen.map(lambda line:line.encode('utf-8').split(','))\
    .filter(lambda line:line[5]!="" and line[12]!="")\
    .map(lambda line:(line[15],(float(line[5]),float(line[12]),1)))\
    .reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1],a[2]+b[2]))\
    .sortByKey()

###（日期,(住院总费用，住院统筹支付，门诊总费用，门诊统筹支出，住院人次，门诊人次))
result=hospital.leftOuterJoin(men_zhen)\
    .map(hospital_menzhen)\
    .sortByKey()

###（日期，(总费用支出，统筹总支出，住院总费用，住院统筹支付，门诊总费用，门诊统筹支出，住院人次，门诊人次，均次住院统筹支付，均次门诊统筹支付））
out=open('output/workerCostByday.csv','w+')
for (key,value)in result.collect():
    if(value[5]==0):
        out.write("%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f,%.2f\n" % (
        key, value[0] + value[2], value[1] + value[3], value[0], value[1], value[2], value[3], value[4], value[5],
        value[1] / value[4], 0))
    else:
        out.write("%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f,%.2f\n"%(key,value[0]+value[2],value[1]+value[3],value[0],value[1],value[2],value[3],value[4],value[5],value[1]/value[4],value[3]/value[5]))
out.close()

