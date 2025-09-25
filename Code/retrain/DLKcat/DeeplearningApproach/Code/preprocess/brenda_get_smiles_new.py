import json
import requests
import multiprocessing as mp
import time
import random
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import pubchempy as pcp

name_smiles = dict()

def fetch_smiles(compound_name):
    retries = 5
    for attempt in range(retries):
        try:
            compounds = pcp.get_compounds(compound_name, 'name')
            if compounds:
                smiles = compounds[0].canonical_smiles
                return compound_name, smiles
            else:
                return compound_name, None
        except Exception as e:
            print(f"Error fetching {compound_name} (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2 ** attempt + random.uniform(0, 1))  # Exponential backoff with jitter
    return compound_name, None

def save_progress(name_smiles, filename):
    with open(filename, 'w') as wf:
        json.dump(name_smiles, wf, indent=2)

def main():
    df_1 = pd.read_csv('../../Data/database/Kcat_brenda_clean.tsv', sep='\t')
    substrates = df_1['Substrate'].tolist()
    names = list(set(substrates))
    print("Number of unique substrates:", len(names))

    # 使用多进程处理
    num_workers = 5
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(fetch_smiles, names), total=len(names)))

    # 收集结果
    for compound_name, smiles in results:
        name_smiles[compound_name] = smiles

    # 保存结果
    save_progress(name_smiles, '../../Data/database/Kcat_brenda_smiles_0925.json')

if __name__ == '__main__':
    main()
