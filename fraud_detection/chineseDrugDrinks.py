#-*-coding:utf-8-*-
'''
中药饮品处理
'''
###
import xlrd

drug_type={}
workbook=xlrd.open_workbook('dataset/常用中药饮片药品目录.xlsx')
sheet=workbook.sheet_by_name('Sheet1')
clos=sheet.col_values(2)
for i in range(1,len(clos)):
    if clos[i] not in drug_type:
        drug_type[clos[i]]='中药饮品'

drug=open("dataset/中药饮品部分.txt")
for input in drug:
    lines=input.split('、')
    for line in lines:
        if line not in drug_type:
            drug_type[line]='中药饮品'

for key,value in drug_type.items():
    print(key,value)