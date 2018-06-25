#coding:utf-8
'''
汇总每种疾病的均次费用支出情况
'''
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from pyspark import SparkContext

####((年份，病种名称)，（总费用，统筹费用支出，人次）)
def hospitalProcessing(line):
    ### 医院等级的划分
    try:
        line=line.split(",")
        for i in range(6,20):
            if line[i]=="":
                line[i]=0
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
            return ((s[0:4],line[27]),(float(line[6]),float(line[17]),1))
        else:
            return (str(999999), 1)
    except Exception:
        return (str(999999), 1)


sc = SparkContext()
######读入病人住院数据
####((年份，病种名称)，（总费用，统筹费用支出，人次,均次总费用，均次统筹费用支出）)
####((年份，均次统筹费用支出)，（总费用，统筹费用支出，人次,均次总费用,病种名称）) 排序
####((年份，病种名称)，（总费用，统筹费用支出，人次,均次总费用，均次统筹费用支出）)
hospital = sc.textFile("/mif/data_new/worker_hospital.txt")
hospital=hospital.map(hospitalProcessing) \
    .filter(lambda (key, value): isinstance(value,int) == False) \
    .reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1],a[2]+b[2]))\
    .map(lambda (key,value):(key,(value[0],value[1],value[2],value[0]*1.0/value[2],value[1]*1.0/value[2])))\
    .map(lambda (key,value):((key[0],value[4]),(value[0],value[1],value[2],value[3],key[1])))\
    .sortByKey(False) \
    .map(lambda (key, value): ((key[0], value[4]), (value[0], value[1], value[2], value[3], key[1]))) \
    .sortByKey()

out = open('output/avghfeesBydisease.csv', 'w+')
#####(诊疗项目名称,使用次数)
for (key,value) in hospital.collect():
    out.write("%s,%s,%.2f,%.2f,%d,%.2f,%.2f\n"%(key[0],key[1],value[0], value[1], value[2], value[3],value[4]))
out.close()
