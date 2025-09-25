#!/usr/bin/python
# coding: utf-8


import requests
import time
from tqdm import tqdm
import random
# import os
# Extract EC number list from ExPASy, which is a repository of information relative to the nomenclature of enzymes.You can get the enzyme data file below at https://ftp.expasy.org/databases/enzyme/

def eclist():# get all EC numbers from the data file as a list
    with open('../../Data/EC_enzyme/enzyme.dat', 'r') as rf :
        lines = rf.readlines()

    ec_list = list()
    for line in lines :
        if line.startswith('ID') :# example:ID   1.1.1.1
            ec = line.strip().split('  ')[1]
            ec_list.append(ec)
    # print(ec_list)
    # print(len(ec_list)) # 7906--8226--8236
    return ec_list

def sabio_info(allEC):# download data from sabio (including not only kcat)
    QUERY_URL = 'http://sabiork.h-its.org/sabioRestWebServices/kineticlawsExportTsv'

    # specify search fields and search terms

    # query_dict = {"ECNumber":'"1.1.1.1"',}
    i = 0
    count = 0
    for EC in tqdm(allEC) :
        EC = EC.strip() # EC contains a space in front
        print(EC)
        i += 1
        # print('This is %d ----------------------------' %i)
        print('Downloading '+EC)
        query_dict = {"ECNumber":'%s' %EC,}
        query_string = ' AND '.join(['%s:%s' % (k,v) for k,v in query_dict.items()])

        # specify output fields and send request

        query = {'fields[]':['EntryID', 'Substrate', 'EnzymeType', 'PubMedID', 'Organism', 'UniprotID','ECNumber','Parameter'], 'q':query_string}

        request = requests.post(QUERY_URL, params = query)
        time.sleep(random.random()*3) # the sleep time here maybe a little long

        if request.text :
            with open('../../Data/database/Kcat_sabio_4/%s.txt' %EC, 'w',encoding='utf-8') as ECfile :
                ECfile.write(request.text)
        else:
            count+=1
    print("there is %d not correctly download" %count)


if __name__ == '__main__' :
    sabio_info(eclist()) # 8236/8236 [10:11:14<00:00,  4.45s/it] succesfully run on linux