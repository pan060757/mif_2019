from pyspark import SparkContext
sc = SparkContext()

from pyspark.mllib.fpm import FPGrowth

data = sc.textFile("/mif/data_new/worker_hospital_detail.txt")
data = data.map(lambda line: line.split(','))
# num 0 ,medical_name 2 ,count 4
data_ngs = data.filter(lambda line: len(line) > 4)
#basket
data_bkt_withNum = data_ngs.map(lambda line: ((line[0], line[2]), 1)) \
    .reduceByKey(lambda a, b: a) \
    .map(lambda (k, v): (k[0], [k[1]])) \
    .reduceByKey(lambda a, b: a + b)

data_bkt=data_bkt_withNum.map(lambda (k, v): v)
data_bkt.cache()
model = FPGrowth.train(data_bkt,minSupport=0.2,numPartitions=10)
fitems = model.freqItemsets().collect()
out = open('output/data.txt', 'w')
for itemset in fitems:
    line = reduce(lambda a, b: "%s\t%s"%(a,b), itemset.items).encode("utf-8")
    out.write("%d\t%s\n" % (itemset.freq,line))
out.close()
