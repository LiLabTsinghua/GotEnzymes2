#!/usr/bin/python
# coding: utf-8
'''
从多个TXT(TSV)文件中读取Kcat和Km，并将特定条件下的数据集合到一个新的TSV文件中。
example:
EntryID	Type	ECNumber	Substrate	EnzymeType	PubMedID	Organism	UniprotID	Value	Unit
1	kcat	3.2.1.103	Gal-beta1->4GlcNAc-beta1->3Gal-beta1->4GlcNAc-beta-pNP	wildtype	12950254	Citrobacter freundii		72.9	s^(-1)
EntryID	Substrate	EnzymeType	PubMedID	Organism	UniprotID	ECNumber	parameter.type	parameter.associatedSpecies	parameter.startValue	parameter.endValue	parameter.standardDeviation	parameter.unit
3804	1-Octanol;NAD+	wildtype class III	2936344	Homo sapiens	P11766	1.1.1.1	Km	1-Octanol	5.5E-4		-	M
'''
# goal:
# EC_number	Type	Value	Maximum	Substrate	Commentary	Organism	LigandStructureId	Literature

import os
import csv
from tqdm import tqdm

with open("../../Data/database/Kcat_sabio_4_unisubstrate__test.tsv", 'w') as wf:
    # with open("./Kcat_sabio.tsv", "wt") as outfile :
    tsv_writer = csv.writer(wf, delimiter="\t")
    tsv_writer.writerow(["EntryID", "Type", "ECNumber", "Substrate", "EnzymeType", "PubMedID", 
        "Organism", "UniprotID", "Value", "Unit"])
    
    filenames = os.listdir('../../Data/database/Kcat_sabio_4')
    print(len(filenames)) # 1741 EC files --8236
    i = 0
    j = 0
    no_km=[]
    total_entry=[]
    for filename in tqdm(filenames) :
        # print("Now "+filename[0:-4])# only remain the ec numbers
        # if filename == '1.1.1.184.txt' :
        if filename != '.DS_Store' :
            with open("../../Data/database/Kcat_sabio_4/%s" % filename, 'r', encoding="utf-8") as rf :
                lines = rf.readlines()
            for line in lines[1:] :# not include the head of table
                data = line.strip().split('\t')
                try :
                    if data[7] == 'kcat' and data[9] :
                        i += 1
                        j += 1
                        entryID = data[0]
                        total_entry.append(entryID)
                        t=0
                        for line in lines[1:] :
                            data2 = line.strip().split('\t')
                            if data2[0] == entryID and data2[7] == 'Km' :# same entryID and have both kcat and km to get the substrate
                                t += 1
                                # j += 1
                                # print(j)
                                # tsv_writer.writerow(["EntryID", "Type", "ECNumber", "Substrate", "EnzymeType", "PubMedID", "Organism", "UniprotID", "Value", "Unit"]) yi yi dui ying, substrate = data2[8]
                                # tsv_writer.writerow([j, data[7], data[6], data2[8], data[2], data[3], data[4], data[5], data[9], data[-1]])
                        if t==0:
                            no_km.append(entryID)
                        tsv_writer.writerow([j, data[7], data[6], data[1], data[2], data[3], data[4], data[5], data[9], data[-1]])
                except :
                    continue
    print('no_km:',len(set(no_km)))
    set_total_entry=set(total_entry)
    print('total_entry:',len(set_total_entry))
    # tsv_writer.writerow(["EntryID", "Type", "ECNumber", "Substrate", "EnzymeType", "PubMedID", 
    #     "Organism", "UniprotID", "Value", "Unit"])