#-*-coding:utf-8 -*-
'''
药品种类匹配
'''
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
drug_type={}
drug=open("dataset/drug.txt")
for line in drug:
    lines=line.strip("\r\n").split(' ')
    name=lines[0]
    type=lines[1]
    drug_type[name]=type

drug_use=open("dataset/medicineSortByPrice.csv",encoding='utf-8')
match_cout=0
unmatch_cout=0
for line in drug_use:
    lines = line.strip("\r\n").split(',')
    name=lines[0]
    for str in ['注射液','胶囊','片','(',')','注射用','颗粒','糖浆','口服溶液']:
        name=name.replace(str, '')
    flag = False
    for key,value in drug_type.items():
       for str in ['注射液', '胶囊', '片', '(', ')', '注射用','颗粒','糖浆','口服溶液']:
            key=key.replace(str, '')
       if fuzz.ratio(key,name)>70:
           match_cout=match_cout+1
           flag=True
           break
    if flag==False:
        print(lines[0])
        unmatch_cout=unmatch_cout+1

print("匹配药品数：",match_cout)
print("未匹配药品数：",unmatch_cout)
drug.close()
drug_use.close()