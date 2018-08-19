# -*- coding:utf-8 -*-
import xlrd
import xlwt
from datetime import date

# 注意在读取时要添加formatting_info=True参数，默认是False，表示原样读取
wb = xlrd.open_workbook('dataset/中成药部分.xls')
workbook=xlwt.Workbook(encoding='ascii')
worksheet=workbook.add_sheet('中成药部分')
####获取一个sheet对象的列表
sheets=wb.sheets()
for table in sheets:
    table = wb.sheet_by_name(table.name)
    # 计算出合并的单元格有哪些
    colspan = {}     # 用于保存计算出的合并的单元格，key=(7, 4)合并单元格坐标，value=(7, 2)合并单元格首格坐标
    # table.merged_cells是一个元组的集合，每个元组由4个数字构成(7，8，2，5)
    # 四个数字依次代表：行，合并的范围(不包含)，列，合并的范围(不包含)，类似range()，从0开始计算
    # (7，8，2，5)的意思是第7行的2,3,4列进行了合并
    # print('table.merged_cells:',str(table.merged_cells))
    if table.merged_cells:
        for item in table.merged_cells:
            # print 'item: ' + str(item)
            # 通过循环进行组合，从而得出所有的合并单元格的坐标
            for row in range(item[0], item[1]):
                for col in range(item[2], item[3]):
                    # 合并单元格的首格是有值的，所以在这里进行了去重
                    if (row, col) != (item[0], item[2]):
                        colspan.update({(row, col): (item[0], item[2])})
    # print(colspan)

    # 开始循环读取excel中的每一行的数据
    for i in range(2,table.nrows-3):
        for j in range(table.ncols):
            # 假如碰见合并的单元格坐标，取合并的首格的值即可
            if colspan.get((i, j)):
                print(table.cell_value(*colspan.get((i, j))),end=' ')
            else:
                print(table.cell_value(i, j),end=' ')
        print('\r')