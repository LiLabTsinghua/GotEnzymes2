#!/usr/bin/python
# coding: utf-8

# Run in python 3.7
'''
这段Python代码主要功能是处理一个TSV文件，该文件包含关于生物化学参数Kcat（一种测量酶速率常数的指标）的数据。代码的目的是清洗这些数据，选择最大的Kcat值，统一单位，并去除重复项，最终将清洗后的数据保存到一个新的TSV文件中。
'''
'''
output example:
Type	ECNumber	Substrate	EnzymeType	PubMedID	Organism	UniprotID	Value	Unit
kcat	3.2.1.103	Gal-beta1->4GlcNAc-beta1->3Gal-beta1->4GlcNAc-beta-pNP	wildtype	12950254	Citrobacter freundii		72.9	s^(-1)
'''
import csv

with open("../../Data/database/Kcat_sabio_4_unisubstratexx.tsv", "r", encoding='utf-8') as rf :
    # lines = file.readlines()[1:].strip('\n')
    lines = rf.readlines()[1:]
    lines = [line for line in lines if line.strip()!='']
    # i=0
    Kcat_data = list()
    Kcat_data_include_value = list()
    for line in lines:
        data = line.strip().split('\t')
        Type = data[1]
        ECNumber = data[2]
        Substrate = data[3]
        EnzymeType = data[4]
        PubMedID = data[5]
        Organism = data[6]
        UniprotID = data[7]
        Value = data[8]
        Unit = data[9]
        Kcat_data_include_value.append([Type, ECNumber, Substrate, EnzymeType, PubMedID, Organism, UniprotID, Value, Unit])
        Kcat_data.append([Type, ECNumber, Substrate, EnzymeType, PubMedID, Organism, UniprotID])

print("Kcat_data",len(Kcat_data))  # 22683 items for not unique substrate --26345

# 去除重复项
new_lines = list()
for line in Kcat_data :
    if line not in new_lines :
        new_lines.append(line)

print("new_lines",len(new_lines))  # 21627 included all elements, 18296 included all except for Kcat value and unit --21507

# i = 0
clean_Kcat = list()
for new_line in new_lines :
    value_unit = dict()
    Kcat_values = list()
    for line in Kcat_data_include_value :
        if line[:-2] == new_line :
            value = line[-2]
            value_unit[str(float(value))] = line[-1]
            # print(type(value))  # <class 'str'>
            Kcat_values.append(float(value))
    # print(value_unit)
    # print(Kcat_values)
    max_value = max(Kcat_values) # choose the maximum one for duplication Kcat value under the same entry as the data what we use
    unit = value_unit[str(max_value)]
    # print(max_value)
    # print(unit)

    if unit in ['mol*s^(-1)*mol^(-1)', 's^(-', '-'] :
        unit = 's^(-1)'
        # print("unit changed")# 29 unit changed in total
    new_line.append(str(max_value))
    new_line.append(unit)
    if new_line[-1] == 's^(-1)' :
        clean_Kcat.append(new_line)


# print(clean_Kcat)
print("clean_Kcat",len(clean_Kcat))  # 18243 after unifing the Kcat value unit to 's^(-1)', in which 16825 has a specific Unipro ID # 21461 --21452


with open("../../Data/database/Kcat_sabio_clean_unisubstratexx.tsv", "w") as wf :
    records = ['Type', 'ECNumber', 'Substrate', 'EnzymeType', 'PubMedID', 'Organism', 'UniprotID', 'Value', 'Unit']
    wf.write('\t'.join(records) + '\n')
    for line in clean_Kcat :
        wf.write('\t'.join(line) + '\n')
