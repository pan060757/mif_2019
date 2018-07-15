#-coding:utf-8-*-
'''
频繁用药情况统计
'''
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from pyspark import SparkContext
from pyspark import StorageLevel

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
            return ((line[1],s),line[2])
        else:
            return (str(999999), 1)
    except Exception:
        return (str(999999), 1)

#####数据预处理(只提取药品名模块)
####((就医序号，药品名称)，次数)
def hospital_detail_processing(line):
    try:
        line = line.encode('utf-8').split(',')
        if len(line) >4 and line[7] != "":
            return ((line[0],line[2]),1)
        else:
            return (999999, 1)
    except Exception:
        return (999999, 1)

#####提取药品数据
def drug_extraction((key,value)):
    drug=""
    values=value[1].split(',')
    for i in range(1,len(values)):
        if values[i] in med_broadcast.value:
            if drug=="":
                drug=values[i]
            else:
                drug=drug+","+values[i]
    return(key,(value[0],drug))

sc = SparkContext()
#####读入药品列表
medicine=sc.textFile('file:///home/edu/songsong/mif_2019/fraud_detection/medicine.txt')
###得到药品目录的列表
medicine=medicine.map(lambda line:line.encode('utf-8').split(','))\
    .map(lambda line:(line[1],1))\
    .sortByKey()
med_List=medicine.keys().collect()
med_broadcast=sc.broadcast(med_List)

#####读入员工住院数据
###(就医序号，三大目录id,药品、诊疗名称)，(单价，数量，类型 ，限价，总费用,使用次数）
data = sc.textFile("/mif/data_new/worker_hospital_detail.txt")
data=data.map(hospital_detail_processing)\
    .filter(lambda(key,value):(isinstance(key,int)==False))\
    .reduceByKey(lambda a,b:a+b)\
    .map(lambda (key,value):(key[0],key[1]))\
    .reduceByKey(lambda a,b:a+','+b)\
    .sortByKey()

# data.repartition(1).saveAsTextFile("file:///home/edu/songsong/mif_2019/fraud_detection/test")

# #####读入腰椎间盘突出患者的数据
yzjptc=sc.textFile('file:///home/edu/songsong/mif_2019/fraud_detection/output/dataOfYZJPTC.csv')
yzjptc=yzjptc.map(dataProcessing)\
    .sortByKey()

######读入病人住院数据
hospital = sc.textFile("/mif/data_new/worker_hospital.txt")
hospital=hospital.map(hospitalProcessing) \
    .filter(lambda (key, value): isinstance(value,int) == False) \
    .sortByKey()

####(（个人编号，入院日期)，（就诊记录，就医序号））
####（就医序号，就诊记录）
yzjptc_hospital=yzjptc.join(hospital)\
    .map(lambda (key,value):(value[1],value[0]))\
    .sortByKey()

result=yzjptc_hospital.join(data) \
    .map(drug_extraction) \
    .sortByKey()

out = open('output/fpm_medicine_analysis.csv', 'w+')
####(使用药品情况)
for (key,value) in result.collect():
    out.write("%s\n"%(value[1]))
out.close()
