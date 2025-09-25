#!/usr/bin/python
# coding: utf-8

import re

'''
从一个已经存在的TSV文件中提取数据，针对特定字段（在本例中是“EnzymeType”字段）进行处理，最终生成一个新的TSV文件，其中包含经过特定处理的数据行。具体来说，代码检查“EnzymeType”字段以识别和提取“wildtype”和突变类型，后者通过正则表达式匹配具体的突变格式（例如“A234B”）。
只考虑突变时的情形，是支线任务
'''

with open("../../Data/database/Kcat_sabio_clean_unisubstratexx.tsv", "r", encoding='utf-8') as rf :
    lines = rf.readlines()[1:]

clean_mutant = list()
for line in lines :
    # print(line)
    data = line.strip().split('\t')
    Type = data[0]
    ECNumber = data[1]
    Substrate = data[2]
    EnzymeType = data[3]
    PubMedID = data[4]
    Organism =data[5]
    UniprotID = data[6]
    Value = data[7]
    Unit = data[8]

    if 'wildtype' in EnzymeType :
        enzymeType = 'wildtype'
    else :
    # if 'mutant' in EnzymeType or 'mutated' in EnzymeType:
        # print(EnzymeType)
        mutant = re.findall('[A-Z]\d+[A-Z]', EnzymeType)  # use re to find string like A234B
        enzymeType = '/'.join(mutant)

    # print(enzymeType)
    if enzymeType :
        clean_mutant.append([Type, ECNumber, Substrate, enzymeType, PubMedID, Organism, UniprotID, Value, Unit])


# print(enzymeType_entries)
print("clean_mutant",len(clean_mutant))  # 17384 --20431 --20421

with open("../../Data/database/Kcat_sabio_clean_unisubstratexx_2.tsv", "w") as wf :
    records = ['Type', 'ECNumber', 'Substrate', 'EnzymeType', 'PubMedID', 'Organism', 'UniprotID', 'Value', 'Unit']
    wf.write('\t'.join(records) + '\n')
    for line in clean_mutant :
        wf.write('\t'.join(line) + '\n')


# with open("../../Data/database/Kcat_sabio_clean_unisubstrate.tsv", "r", encoding='utf-8') as file :
#     lines = file.readlines()[1:]

# enzymeTypes = [line.strip().split('\t')[3] for line in lines]

# print(len(enzymeTypes)) # 18243

# enzymeType_entries = list()
# for desc in enzymeTypes :
#     if 'wildtype' in desc :
#         enzymeType = 'wildtype'
#     else :
#     # if 'mutant' in desc or 'mutated' in desc:
#         print(desc)
#         mutant = re.findall('[A-Z]\d+[A-Z]', desc)  # re is of great use
#         if len(mutant) >=1 :
#             enzymeType = '/'.join(mutant)

#     if enzymeType :
#         enzymeType_entries.append(enzymeType)

# # print(enzymeType_entries)
# print(len(enzymeType_entries))  
