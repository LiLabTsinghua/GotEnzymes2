#!/usr/bin/python
# coding: utf-8

# Author: LE YUAN
# Date: 2020-07-23

import json
# import time
import requests
import multiprocessing as mp
from multiprocessing.dummy import Pool
from tqdm import tqdm
# from pubchempy import Compound, get_compounds


name_smiles = dict()

'''
obtain canonical SMILES just by chemical name using PubChem API
obtain SMILES by PubChem API using the website
使用PubChem API根据化学品名称获取它们的标准SMILES表示。该脚本利用Python的多进程能力来加速API请求。
'''
def get_smiles(name):
    # smiles = redis_cli.get(name)
    # if smiles is None:
    try :
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT' % name
        req = requests.get(url)
        if req.status_code != 200:
            smiles = None
        else:
            smiles = req.content.splitlines()[0].decode()
            # print(smiles)
        # redis_cli.set(name, smiles, ex=None)

        # print smiles
    except :
        smiles = None

    name_smiles[name] = smiles

# To obtain SMILES for substrates using provided API by PubChem
# def main():
#     # with open('./smiles_data.json') as f:
#     #     names = json.load(f)
#     #     print(len(names))

#     with open("../../Data/database/Kcat_sabio_clean_unisubstrate99.tsv", "r", encoding='utf-8') as rf:
#         lines = rf.readlines()[1:]

#     substrates = [line.strip().split('\t')[2] for line in lines]

#     print("substrates",len(substrates)) # 18243 --21461 --21439

#     names = list(set(substrates))
#     print("names",len(names))  # 3100 --3273 --3273

#     # for substrate in substrates[:100] :
#     #     print(substrate)

#     # thread_pool = mp.Pool(4)
#     thread_pool = Pool(4)
#     tqdm(thread_pool.map(get_smiles, names))

#     with open('../../Data/database/Kcat_sabio_smiles.json', 'w') as wf:
#         json.dump(name_smiles, wf, indent=2)
from tqdm import tqdm

def process_items(names):
    with Pool(4) as pool:
        for _ in tqdm(pool.imap_unordered(get_smiles, names), total=len(names)):
            pass

def main():
    with open("../../Data/database/Kcat_sabio_clean_unisubstrate.tsv", "r", encoding='utf-8') as rf:
        lines = rf.readlines()[1:]

    substrates = [line.strip().split('\t')[2] for line in lines]
    names = list(set(substrates))

    print("Number of unique substrates:", len(names))

    process_items(names)

    with open('../../Data/database/Kcat_sabio_smiles.json', 'w') as wf:
        json.dump(name_smiles, wf, indent=2)
if __name__ == '__main__':
    main()

# Small example: 
# results = get_compounds('aspirin', 'name')
# for compound in results :
#     print(compound.canonical_smiles)

# have a try by running 100 case
# with open("../../Data/database/Kcat_sabio_clean_unisubstrate.tsv", "r", encoding='utf-8') as file :
#     lines = file.readlines()[1:]
# substrates = [line.strip().split('\t')[2] for line in lines]

# print(len(substrates))
# print(substrates[:10])

# for substrate in substrates[:100] :
#     print(substrate)
#     results = get_compounds(substrate, 'name')
#     print(len(results))
#     if len(results) >0 :
#         print(results[0].canonical_smiles)
#     else :
#         print('-------------------------------------------------')
    # # To test how many entries having SMILES for Sabio-RK database
# def main():
#     with open('../../Data/database/Kcat_sabio_smiles.json', 'r') as infile:
#         name_smiles = json.load(infile)

#     with open("../../Data/database/Kcat_sabio_clean_unisubstrate.tsv", "r", encoding='utf-8') as file :
#         lines = file.readlines()[1:]

#     substrates = [line.strip().split('\t')[2] for line in lines]

#     print(len(substrates)) # 18243

#     substrate_smiles = list()
#     for substrate in substrates :
#         # print(substrate)
#         smiles = name_smiles[substrate]
#         # print(smiles)
#         if smiles is not None :
#             # print(smiles)
#             substrate_smiles.append(smiles)

#     print(len(substrate_smiles))  # 15218 have SMILES
# Another method to retrieve SMILES by Pubchempy 
# def get_smiles(name):
#     time.sleep(0.5)
#     results = get_compounds(name, 'name')

#     # print(len(results))
#     if len(results) >0 :
#         smiles = results[0].canonical_smiles
#         print(smiles)
#     else :
#         smiles = None
#         print(smiles)
#         print('-------------------------------------------------')

#     name_smiles[name] = smiles