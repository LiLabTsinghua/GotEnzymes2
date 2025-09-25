#!/usr/bin/python
# coding: utf-8

# Author: LE YUAN
# Date: 2020-10-03

import math
import json
import pickle
import numpy as np
import torch
from collections import defaultdict
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import os
word_dict = defaultdict(lambda: len(word_dict))
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

proteins = list()
compounds = list()
adjacencies = list()
regression =list()

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    # print(sequence)
    words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(words)
    # return word_dict

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    # atom_dict = defaultdict(lambda: len(atom_dict))
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    # print(atoms)
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    # bond_dict = defaultdict(lambda: len(bond_dict))
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    # fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    # edge_dict = defaultdict(lambda: len(edge_dict))

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def tensor_long(array_data):
    tensor_list = [torch.tensor(sublist, dtype=torch.long) for sublist in array_data]
    return tensor_list
def tensor_float(array_data):
    tensor_list = [torch.tensor(sublist, dtype=torch.float) for sublist in array_data]
    return tensor_list

    # Save the list of tensors using pickle
    # with open('+array_data+'.pkl', 'wb') as f:
    #     pickle.dump(tensor_list, f)

    # Load the list of tensors using pickle
    # with open('tensor_list.pkl', 'rb') as f:
    #     loaded_tensor_list = pickle.load(f)
    # print(loaded_tensor_list)


def main() :
    # with open('../../Data/database/Kcat_combination_0918_wildtype_mutant.json', 'r') as infile :
    Kcat_data = pd.read_csv('../../Data/Test_kcat_prottrans_log2.csv')

    # print(len(Kcat_data))

    # radius = 3 # The initial setup, I suppose it is 2, but not 2.
    radius = 2
    ngram = 3

    """Exclude data contains '.' in the SMILES format."""
    i = 0
    # for data in Kcat_data :
    for index, row in tqdm(Kcat_data.iterrows(), total=Kcat_data.shape[0]):
        smiles = row['Smiles']
        sequence = row['Sequence']
        Kcat = row['Value']
        # smiles = data['Smiles']
        # sequence = data['Sequence']
        # print(smiles)
        # Kcat = data['Value']
        if "." not in smiles and float(Kcat) > 0:
            # i += 1
            # print('This is',i)
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol)
            # print(atoms)
            i_jbond_dict = create_ijbonddict(mol)
            # print(i_jbond_dict)

            fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
            # print(fingerprints)
            compounds.append(fingerprints)

            adjacency = create_adjacency(mol)
            adjacencies.append(adjacency)

            words = split_sequence(sequence,ngram)
            # print(words)
            proteins.append(words)

            # print(float(Kcat))

            regression.append(np.array([math.log2(float(Kcat))]))
            # print(math.log2(float(Kcat)))

            # regression.append(np.array([math.log10(float(Kcat))]))
            # print(math.log10(float(Kcat)))
        else:
            tqdm.write(smiles)
            # print(Kcat)
            continue
    # np.save('+'compounds', compounds)
    # np.save('+'adjacencies', adjacencies)
    # np.save('+'regression', regression)
    # np.save('+'proteins', proteins)

    # arr = np.asarray(compounds, dtype = object)
    # np.save('+tensor__(compounds), arr)
    # arr = np.asarray(adjacencies, dtype = object)
    # np.save('+'adjacencies', arr)
    # arr = np.asarray(regression, dtype = object)
    # np.save('+'regression', arr)
    # arr = np.asarray(proteins, dtype = object)
    # np.save('+'proteins', arr)
    saving_path = '../../Data/xw_input_kkm_test/'
    if os.path.exists(saving_path) == False:
        os.makedirs(saving_path)
    with open(saving_path+'compounds'+'.pkl', 'wb') as f:
        pickle.dump(tensor_long(compounds), f)
    with open(saving_path+'adjacencies'+'.pkl', 'wb') as f:
        pickle.dump(tensor_float(adjacencies), f)
    with open(saving_path+'regression'+'.pkl', 'wb') as f:
        pickle.dump(tensor_float(regression), f)
    with open(saving_path+'proteins'+'.pkl', 'wb') as f:
        pickle.dump(tensor_long(proteins), f)
    dump_dictionary(fingerprint_dict, saving_path + 'fingerprint_dict.pickle')
    dump_dictionary(atom_dict, saving_path + 'atom_dict.pickle')
    dump_dictionary(bond_dict, saving_path + 'bond_dict.pickle')
    dump_dictionary(edge_dict, saving_path + 'edge_dict.pickle')
    dump_dictionary(word_dict, saving_path + 'sequence_dict.pickle')

if __name__ == '__main__' :
    main()
