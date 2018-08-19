#-coding:utf-8-*-
'''
西药药品种类划分
'''

drug_type={}  ###(药品编号，药品种类)
drug=open('dataset/westernDrugType.txt')
for line in drug:
    lines=line.strip('\r\n').split(',')
    no=lines[0]
    name=lines[1]
    drug_type[no]=name

data_input=open('dataset/西药部分.txt')
####记住之前的数值
flag=""
for line in data_input:
    lines=line.strip('\r\n').split(' ')
    no=lines[0]
    name=lines[7]
    if(len(no)==2):
        flag=drug_type[no]
    if(no==""):
        print(name,flag)