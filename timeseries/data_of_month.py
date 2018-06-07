#-*-coding:utf-8-*-
'''
构造月份数据集
'''

# load dataset
# 加载数据
from pandas import read_csv

data_input= read_csv('dataset/workerCostByMonth.csv', header=None)
#####
data_input.columns=['date','total_fees','group_fees','hospital_fees','h_groupfees','menzhen_fees','m_groupfees','hospital_count','menzhen_count','avg_hgroupfees','avg_mgroupfees']
#####只提取一个特征
values=data_input['group_fees']
count=0
out=open("dataset/data_of_month.csv","w")
for start in range(0,len(values)-12):
    index = 0
    for i in range(1,12):
        line = []
        line.append(values[start])
        for j in range(1,i):
            index=start+j
            line.append(values[index])
        if(index>start):
            line.append(values[index+1])
            line.append(values[index+2])
        else:
            line.append(values[start + 1])
            line.append(values[start+ 2])
        count=count+1
        for k in range(len(line)-1):
            out.write(str(line[k]))
            out.write(",")
        out.write(str(line[len(line)-1]))
        out.write("\n")
print(count)
out.close()
