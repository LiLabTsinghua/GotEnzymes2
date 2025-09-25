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

word_dict = defaultdict(lambda: len(word_dict))
atom_dict = defaultdict(lambda: len(atom_dict))
# atom_dict = {}
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

proteins = list()
compounds = list()
smilesadjacencies = list()
regression =list()
proteinadjacencies = list()
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
                try:
                    fingerprints.append(fingerprint_dict[fingerprint])
                except:
                    fingerprint_dict[fingerprint] = 0
                    fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    try:
                        edge = edge_dict[(both_side, edge)]
                    except:
                        edge_dict[(both_side, edge)] = 0
                        edge = edge_dict[(both_side, edge)]

                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)
def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)
def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)
def tensor_long(array_data):
    tensor_list = [torch.tensor(sublist, dtype=torch.long) for sublist in array_data]
    return tensor_list
def tensor_float(array_data):
    tensor_list = [torch.tensor(sublist, dtype=torch.float) for sublist in array_data]
    return tensor_list


def get_ca_coords(pdb):
    with open(pdb, 'r') as file:
        lines = file.readlines()
        file.close()

    out = []

    for line in lines:
        if line.startswith('ATOM ') and line.split()[4] == 'A' and line.split()[2] == 'CA':
            res_num = line.split()[5]
            res_name = line.split()[3]
            x = line.split()[6]
            y = line.split()[7]
            z = line.split()[8]
            if len(x) > int(8):
                x = line.split()[6][:-8]
                y = line.split()[6][-8:]
                z = line.split()[7]
            elif len(y) > int(8):
                x = line.split()[6]
                y = line.split()[7][:-8]
                z = line.split()[7][-8:]
            elif len(res_num) > int(4):
                x = line.split()[5][-8:]
                y = line.split()[6]
                z = line.split()[7]
                res_num = line.split()[5][:-8]

            out.append([res_num, res_name, x, y, z])

    df = pd.DataFrame(out, columns=['res_num', 'res_name', 'x', 'y', 'z'])

    return df
def luciferase_contact_map(pdb,seq):
    ca_coords = get_ca_coords(pdb)
    dist_arr = pairwise_distances(ca_coords[['x', 'y', 'z']].values)  # distance
    dist_tensor = torch.from_numpy(dist_arr)
    dist_thres = 10
    cont_arr = (dist_arr < dist_thres).astype(int)
    cont_tensor = torch.from_numpy(cont_arr)
    if cont_arr.shape[0] == len(seq):
        proteinadjacency = sparse.csr_matrix(cont_arr)
    else:
        a = np.zeros((cont_arr.shape[0], len(seq) - cont_arr.shape[0]))
        cont_arr = np.column_stack((cont_arr, a))
        b = np.zeros((len(seq) - cont_arr.shape[0], len(seq)))
        cont_arr = np.row_stack((cont_arr, b))
        row, col = np.diag_indices_from(cont_arr)
        cont_arr[row, col] = 1
        proteinadjacency = sparse.csr_matrix(cont_arr)
    return proteinadjacency
def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    # print(sequence)
    words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(words)

def main() :
    # with open('../../Data/database/Kcat_combination_0918_wildtype_mutant.json', 'r') as infile :
    Kcat_data = pd.read_csv('/home/wuke/project/bio_deeplearning/zzz_benchmark/data/EITLEM_KCAT.csv')
    # Kcat_data = Kcat_data.head(3)
    # radius = 3 # The initial setup, I suppose it is 2, but not 2.
    radius = 2
    ngram = 3

    """Exclude data contains '.' in the SMILES format."""
    i = 0
    # for data in Kcat_data :
    fasta_file_path = '/home/wuke/project/bio_deeplearning/zzz_benchmark/data/bingxue_seq.fasta'
    sequence_to_index = {}
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        seq_id = record.id.split('_')[-1]
        sequence = str(record.seq)
        sequence_to_index[sequence] = seq_id
    Kcat_data['Index'] = None
    for index, row in tqdm(Kcat_data.iterrows(), total=Kcat_data.shape[0]):
        smiles = row['Smiles']
        sequence = row['Sequence']
        Kcat = row['Value']
        if sequence in sequence_to_index:
            Kcat_data.at[index, 'Index'] = sequence_to_index[sequence]
            pdb_filename = f'seq_{sequence_to_index[sequence]}.pdb'
            structure_path = os.path.join('/home/wuke/project/bio_deeplearning/zzz_benchmark/data/bingxue_pdb', pdb_filename)
            if not os.path.exists(structure_path):
                print(f"Warning: No matching structure found for entry at index {sequence_to_index[sequence]}")
                structure_path = None
        else:
            print(f"Error: No matching sequence found for entry at seq {index+1}")
            print(sequence)
            structure_path = None
        if "." not in smiles and float(Kcat) > 0 and structure_path is not None:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict)
            compounds.append(fingerprints)
            smilesadjacency = create_adjacency(mol)
            smilesadjacencies.append(smilesadjacency)
            words = split_sequence(sequence, ngram)
            # print(1, len(words))
            # print(2, len(sequence))
            proteins.append(words)
            proteinadjacency = luciferase_contact_map(structure_path, sequence)
            proteinadjacencies.append(proteinadjacency)
            regression.append(np.array([math.log2(float(Kcat))]))
    # shapes = [arr.shape for arr in smilesadjacencies]
    # unique_shapes = set(shapes)

    # if len(unique_shapes) > 1:
    #     print("Detected multiple shapes:", unique_shapes)
    # else:
    #     print("All elements have the same shape.")
    dump_dictionary(fingerprint_dict, '../../Data/Input/fingerprint_dict.pickle')
    dump_dictionary(atom_dict, '../../Data/Input/atom_dict.pickle')
    dump_dictionary(bond_dict, '../../Data/Input/bond_dict.pickle')
    dump_dictionary(edge_dict, '../../Data/Input/edge_dict.pickle')
    dump_dictionary(word_dict, '../../Data/Input/word_dict.pickle')
    # with open('../../Data/Input/'+'fingerprint'+'.pkl', 'wb') as f:
    #     pickle.dump(tensor_long(compounds), f)
    # with open('../../Data/Input/'+'smilesadjacencies'+'.pkl', 'wb') as f:
    #     pickle.dump(tensor_float(adjacencies), f)
    # with open('../../Data/Input/'+'regression'+'.pkl', 'wb') as f:
    #     pickle.dump(tensor_float(regression), f)
    # with open('../../Data/Input/'+'proteins'+'.pkl', 'wb') as f:
    #     pickle.dump(tensor_long(proteins), f)
    with open('../../Data/Input/proteinadjacencies.pkl', 'wb') as f:
        pickle.dump(proteinadjacencies, f)
    np.save('../../Data/Input/smilesadjacencies.npy', smilesadjacencies)
    np.save('../../Data/Input/regression.npy', regression)
    np.save('../../Data/Input/sequences.npy', proteins)
    np.save('../../Data/Input/fingerprint.npy', compounds)
    # np.save('../../Data/Input/proteinadjacencies.npy', proteinadjacencies)

    # print(smilesadjacencies.shape, regression.shape, proteins.shape, compounds.shape, proteinadjacencies.shape)

if __name__ == '__main__':
    main()