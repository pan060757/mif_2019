#-coding:utf-8
'''相同药品，相同诊疗手段进行合并
'''

from pyspark import SparkContext
###(就医序号，三大目录id,药品、诊疗名称)，(单价，数量，类型 ，限价，总费用,使用次数）
def hospital_processing(line):
    line=line.encode("utf-8").split(',')
    if(len(line)==8):
        for i in range(3, 8):
            if line[i] == "":
                line[i] = '0'
        return ((line[0], line[1], line[2]), (line[3], float(line[4]), line[5], line[6], float(line[7]), 1))
    else:
        return (str(999999),1)

sc = SparkContext()
###(就医序号，三大目录id,药品、诊疗名称)，(单价，数量，类型 ，限价，总费用,使用次数）
data = sc.textFile("/mif/data_new/worker_hospital_detail.txt")
data=data.map(hospital_processing)\
    .filter(lambda(key,value):(isinstance(value,int)==False))\
    .reduceByKey(lambda a,b:(a[0],a[1]+b[1],a[2],a[3],a[4]+b[4],a[5]+b[5]))\
    .sortByKey()

####把数据合并到一个分区中
data.repartition(1).saveAsTextFile("output/worker_hospital_detail_processing.csv")

