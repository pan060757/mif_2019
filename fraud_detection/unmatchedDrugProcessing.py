#coding:utf-8
'''
对未匹配的药品进行处理
'''
import re
input=open("dataset/unmatchDrug.txt")
for line in input:
    lines=line.strip("\r\n").split("\t")
    if(len(lines)>4):
        drugname=lines[2]
        print(drugname)
        drugType=lines[4]
        print(re.split('\d',drugType))