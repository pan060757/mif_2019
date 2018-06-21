#-*-coding:utf-8 -*-
'''
构造病人-使用药品情况数据集
'''
import datetime
import re

from pyspark import SparkContext

####(（个人编号，入院日期)，就诊记录）
def dataProcessing(line):
    lines=line.encode("utf-8").split(",")
    return ((lines[0], lines[13]), line)

####(（个人编号，入院日期)，就医序号）
def hospitalProcessing(line):
    ### 医院等级的划分
    try:
        line=line.split(",")
        if(line[21]!="" and line[22]!=""):       ###可能存在未激励出院时间和住院时间的住院记录
            inHospital = line[21]
            outHospital = line[22]
            s = inHospital.strip("").split('-')
            t = outHospital.strip("").split('-')
            s[1] = re.sub("\D", "", s[1])  ##提取其中数字部分
            t[1] = re.sub("\D", "", t[1])  ##提取其中数字部分
            if len(s[1]) < 2:
                s[1] = '0' + s[1]
            if len(s[0]) < 2:
                s[0] = '0' + s[0]
            if len(t[1]) < 2:
                t[1] = '0' + t[1]
            if len(t[0]) < 2:
                t[0] = '0' + t[0]
            s = '20' + s[2] + s[1] + s[0]     ####入院日期
            return ((line[1],s),line[0])
        else:
            return (str(999999), 1)
    except Exception:
        return (str(999999), 1)

sc = SparkContext()
#####读入腰椎间盘突出患者的数据
data=sc.textFile('file:///home/edu/songsong/mif_2019/fraud_detection/output/dataOfYZJPTC.csv')
data=data.map(dataProcessing)\
    .sortByKey()

######读入病人住院数据
hospital = sc.textFile("/mif/data_new/worker_hospital.txt")
hospital=hospital.map(hospitalProcessing) \
    .filter(lambda (key, value): isinstance(value,int) == False) \
    .sortByKey()

####(（个人编号，入院日期)，（就诊记录，就医序号））
####（就医序号，就诊记录）
data_hospital=data.join(hospital)\
    .map(lambda (key,value):(value[0],key[0]))\
    .sortByKey()

####读取病人使用药品情况

