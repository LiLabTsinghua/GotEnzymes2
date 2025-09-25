import math
import json
import pickle
import numpy as np
import torch
from collections import defaultdict
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sparse
from sklearn.metrics import pairwise_distances
from Bio import SeqIO
import os
import argparse

def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge_key = (both_side, edge)
                    _i_jedge_dict[i].append((j, edge_dict[edge_key]))
            i_jedge_dict = _i_jedge_dict
    return np.array(fingerprints)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def tensor_long(array_data):
    return [torch.tensor(sublist, dtype=torch.long) for sublist in array_data]

def tensor_float(array_data):
    return [torch.tensor(sublist, dtype=torch.float) for sublist in array_data]

def get_ca_coords(pdb):
    out = []
    with open(pdb, 'r') as file:
        for line in file:
            if line.startswith('ATOM '):
                atom_name = line[12:16].strip()
                if atom_name != 'CA':
                    continue
                chain_id = line[21]
                if chain_id != 'A':
                    continue  # 假设仅处理链A
                res_num = line[22:26].strip()
                res_name = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                out.append([res_num, res_name, x, y, z])
    return pd.DataFrame(out, columns=['res_num', 'res_name', 'x', 'y', 'z'])

def luciferase_contact_map(pdb, seq):
    try:
        ca_coords = get_ca_coords(pdb)
        if ca_coords.empty:
            return None
        dist_arr = pairwise_distances(ca_coords[['x', 'y', 'z']].values.astype(float))
        cont_arr = (dist_arr < 10).astype(int)
        if cont_arr.shape[0] == len(seq):
            return sparse.csr_matrix(cont_arr)
        else:
            pad_rows = len(seq) - cont_arr.shape[0]
            cont_arr = np.pad(cont_arr, ((0, pad_rows), (0, pad_rows)), mode='constant')
            np.fill_diagonal(cont_arr, 1)
            return sparse.csr_matrix(cont_arr)
    except Exception as e:
        print(f"Error generating contact map for {pdb}: {e}")
        return None

def split_sequence(sequence, ngram, word_dict):
    sequence = '-' + sequence + '='
    return np.array([word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)])

def process_data(Kcat_data, cv, word_dict, atom_dict, bond_dict, fingerprint_dict, edge_dict):
    proteins, compounds, smilesadjacencies, regression, proteinadjacencies = [], [], [], [], []
    fasta_path = '../../../../data/bingxue_seq.fasta'
    sequence_to_index = {str(record.seq): record.id.split('_')[-1] for record in SeqIO.parse(fasta_path, "fasta")}
    
    for index, row in tqdm(Kcat_data.iterrows(), total=Kcat_data.shape[0], desc=f'Processing CV{cv}'):
        try:
            smiles = row['Smiles']
            sequence = row['Sequence']
            Kcat = row['Value']
            if '.' in smiles or float(Kcat) <= 0:
                continue
            
            seq_id = sequence_to_index.get(sequence, None)
            if not seq_id:
                print(f"Sequence not found for index {index}")
                continue
            
            pdb_path = f'../../../../data/bingxue_pdb/seq_{seq_id}.pdb'
            if not os.path.exists(pdb_path):
                # print(f"PDB not found: {pdb_path}")
                continue
            
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if not mol:
                print(f"Invalid SMILES: {smiles}")
                continue
            
            atoms = create_atoms(mol, atom_dict)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(atoms, i_jbond_dict, 2, fingerprint_dict, edge_dict)
            adjacency = create_adjacency(mol)
            
            words = split_sequence(sequence, 3, word_dict)
            contact_map = luciferase_contact_map(pdb_path, sequence)
            if contact_map is None:
                continue
            
            compounds.append(fingerprints)
            smilesadjacencies.append(adjacency)
            proteins.append(words)
            proteinadjacencies.append(contact_map)
            regression.append(np.array([math.log2(float(Kcat))]))
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    return proteins, compounds, smilesadjacencies, regression, proteinadjacencies

def main():
    parser = argparse.ArgumentParser(description='Process Kcat data for specified CV fold.')
    parser.add_argument('--cv', type=int, required=True, help='Cross-validation fold number (1-4)')
    args = parser.parse_args()
    cv = args.cv

    # 初始化字典
    word_dict = defaultdict(lambda: len(word_dict))
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    # 数据加载
    data_path = f'../../../../data/cv/{cv}/EITLEM_KCAT.csv'
    Kcat_data = pd.read_csv(data_path)
    # 数据处理
    proteins, compounds, smilesadj, reg, prot_adj = process_data(
        Kcat_data, cv, word_dict, atom_dict, bond_dict, fingerprint_dict, edge_dict
    )

    # 创建输出目录
    output_dir = f'../../Data/Input/cv{cv}/'
    os.makedirs(output_dir, exist_ok=True)

    # 保存字典
    dicts = {
        'word_dict.pickle': word_dict,
        'atom_dict.pickle': atom_dict,
        'bond_dict.pickle': bond_dict,
        'fingerprint_dict.pickle': fingerprint_dict,
        'edge_dict.pickle': edge_dict
    }
    for name, d in dicts.items():
        dump_dictionary(d, os.path.join(output_dir, name))

    # 保存数据
    with open(os.path.join(output_dir, 'smilesadjacencies.pkl'), 'wb') as f:
        pickle.dump(smilesadj, f)
    with open(os.path.join(output_dir, 'regression.pkl'), 'wb') as f:
        pickle.dump(reg, f)
    with open(os.path.join(output_dir, 'sequences.pkl'), 'wb') as f:
        pickle.dump(proteins, f)
    with open(os.path.join(output_dir, 'fingerprint.pkl'), 'wb') as f:
        pickle.dump(compounds, f)
    with open(os.path.join(output_dir, 'proteinadjacencies.pkl'), 'wb') as f:
        pickle.dump(prot_adj, f)

if __name__ == '__main__':
    main()