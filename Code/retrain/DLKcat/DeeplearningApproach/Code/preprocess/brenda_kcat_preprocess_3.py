#!/usr/bin/python
# coding: utf-8

# This script is to extract Kcat data from all EC files into one file.

import os
import csv
import re

with open("../../Data/database/Kcat_brenda.tsv", "wt") as outfile :
# with open("./Kcat_sabio.tsv", "wt") as outfile :
    tsv_writer = csv.writer(outfile, delimiter="\t")
    tsv_writer.writerow(["EntryID", "Type", "ECNumber", "Substrate", 'EnzymeType', "Organism","Value", "Unit"])
    #                       i,      'kcat', filename[2:-4], data[3],  enzymeType,  data[1], str(value),'s^(-1)'
    filenames = os.listdir('../../Data/database/Kcat_brenda')

    i = 0
    j = 0
    m = 0
    for filename in filenames :
        # print(filename[2:-4])
        if filename != '.DS_Store' :
            with open("../../Data/database/Kcat_brenda/%s" %(filename), 'r', encoding="utf-8") as file :
                lines = file.readlines()

        for line in lines: #-------------------------------------------------------------------error
            data = line.strip().split('\t')
            desc = data[4]#should be describe--close to commentary
            value = float(data[2])
            if value > 0 :  # Kcat value should not be less than 0, but there exist some weird values, less than 0.
                i += 1
                if 'mutant' in desc or 'mutated' in desc:
                    mutant = re.findall(r'[A-Z]\d+[A-Z]', desc)  # re is of great use
                    if len(mutant) >=1 :#多变异
                        enzymeType = '/'.join(mutant)
                    else:
                        i -= 1
                        continue
                else :
                    enzymeType = 'wildtype'
                    m+=1
                tsv_writer.writerow([i, 'kcat', filename[2:-4], data[3], enzymeType, data[1], data[2], 's^(-1)'])
            else:
                j+=1
print('j:',j)#1459  @240820
print('m:',m)#53513 @240820
print('i:',i)#80540 @240820
'''
从所有(很多) EC 文件中提取 Kcat 数据并将其保存到Kcat_brenda.tsv文件中
kcat_brenda.tsv to kcat_brenda.tsv
处理每个 EC 文件：
对于每个 EC 文件，打开文件并读取内容。
遍历文件中的每一行，提取所需的信息（EC 号码，底物，酶类型等）。
如果 Kcat 值大于 0，则将其写入输出文件中。
而且标注了变异与否！！！！！
modefied
'''