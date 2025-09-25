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

class DataPreprocessor:
    def __init__(self, radius=2, ngram=3):
        # Dictionaries should be instance attributes, not global
        self.word_dict = defaultdict(lambda: len(self.word_dict))
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))
        
        self.radius = radius
        self.ngram = ngram

    def split_sequence(self, sequence):
        sequence = '-' + sequence + '='
        words = [self.word_dict[sequence[i:i+self.ngram]] for i in range(len(sequence)-self.ngram+1)]
        return np.array(words)

    def create_atoms(self, mol):
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms)

    def create_ijbonddict(self, mol):
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict

    def extract_fingerprints(self, atoms, i_jbond_dict):
        if (len(atoms) == 1) or (self.radius == 0):
            fingerprints = [self.fingerprint_dict[a] for a in atoms]
        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(self.radius):
                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(self.fingerprint_dict[fingerprint])
                nodes = fingerprints

                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = self.edge_dict[(both_side, edge)]
                        _i_jedge_dict[i].append((j, edge))
                i_jedge_dict = _i_jedge_dict
        return np.array(fingerprints)

    def create_adjacency(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency)

    def dump_dictionaries(self, path):
        """Save all dictionaries to the specified path."""
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'fingerprint_dict.pickle'), 'wb') as file:
            pickle.dump(dict(self.fingerprint_dict), file)
        with open(os.path.join(path, 'atom_dict.pickle'), 'wb') as file:
            pickle.dump(dict(self.atom_dict), file)
        with open(os.path.join(path, 'bond_dict.pickle'), 'wb') as file:
            pickle.dump(dict(self.bond_dict), file)
        with open(os.path.join(path, 'edge_dict.pickle'), 'wb') as file:
            pickle.dump(dict(self.edge_dict), file)
        with open(os.path.join(path, 'sequence_dict.pickle'), 'wb') as file:
            pickle.dump(dict(self.word_dict), file)

    def load_dictionaries(self, path):
        """Load dictionaries from the specified path. This makes dictionaries fixed."""
        with open(os.path.join(path, 'fingerprint_dict.pickle'), 'rb') as file:
            self.fingerprint_dict = defaultdict(int, pickle.load(file))
        with open(os.path.join(path, 'atom_dict.pickle'), 'rb') as file:
            self.atom_dict = defaultdict(int, pickle.load(file))
        with open(os.path.join(path, 'bond_dict.pickle'), 'rb') as file:
            self.bond_dict = defaultdict(int, pickle.load(file))
        with open(os.path.join(path, 'edge_dict.pickle'), 'rb') as file:
            self.edge_dict = defaultdict(int, pickle.load(file))
        with open(os.path.join(path, 'sequence_dict.pickle'), 'rb') as file:
            self.word_dict = defaultdict(int, pickle.load(file))
        
        # When loading, we use defaultdict(int, ...) to ensure new keys get 0 by default, 
        # or handle them as 'unknown'. If you want them to keep incrementing,
        # you'd need to manually set the default factory after loading.
        # For simplicity and to ensure consistency with existing keys, 
        # we'll make sure the `len` based default factory is reactivated.
        # However, for testing, it's safer if unseen items get a specific ID or raise an error.
        # For this problem, if new items appear in test, they should get new IDs, 
        # so re-setting lambda len() is fine.
        max_atom_id = max(self.atom_dict.values()) + 1 if self.atom_dict else 0
        self.atom_dict.default_factory = lambda: max_atom_id + len(self.atom_dict) - max_atom_id # This will make new items get IDs after the loaded ones

        max_word_id = max(self.word_dict.values()) + 1 if self.word_dict else 0
        self.word_dict.default_factory = lambda: max_word_id + len(self.word_dict) - max_word_id
        
        max_bond_id = max(self.bond_dict.values()) + 1 if self.bond_dict else 0
        self.bond_dict.default_factory = lambda: max_bond_id + len(self.bond_dict) - max_bond_id

        max_fingerprint_id = max(self.fingerprint_dict.values()) + 1 if self.fingerprint_dict else 0
        self.fingerprint_dict.default_factory = lambda: max_fingerprint_id + len(self.fingerprint_dict) - max_fingerprint_id

        max_edge_id = max(self.edge_dict.values()) + 1 if self.edge_dict else 0
        self.edge_dict.default_factory = lambda: max_edge_id + len(self.edge_dict) - max_edge_id


    def fit_dictionaries(self, data_path):
        """
        Builds dictionaries by iterating through the dataset.
        This function does not store processed data, only populates dictionaries.
        Typically called on the training data.
        """
        print(f"Building dictionaries from {data_path}...")
        Kcat_data = pd.read_csv(data_path)
        for index, row in tqdm(Kcat_data.iterrows(), total=Kcat_data.shape[0]):
            smiles = row['Smiles']
            sequence = row['Sequence']
            Kcat = row['Value']

            if "." not in smiles and float(Kcat) > 0:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                if mol is None: # Handle invalid SMILES
                    tqdm.write(f"Warning: Invalid SMILES skipped for dictionary building: {smiles}")
                    continue
                
                # These calls will populate the dictionaries as a side effect
                self.create_atoms(mol)
                i_jbond_dict = self.create_ijbonddict(mol)
                self.extract_fingerprints(self.create_atoms(mol), i_jbond_dict) # Pass mol for atoms and i_jbond_dict
                self.split_sequence(sequence)
            else:
                tqdm.write(f"Skipping entry (invalid SMILES or Kcat<=0) for dictionary building: SMILES={smiles}, Kcat={Kcat}")
        print("Dictionary building complete.")

    def transform_data(self, data_path, output_dir):
        """
        Transforms the dataset using the existing dictionaries and saves the output.
        """
        print(f"Transforming data from {data_path} and saving to {output_dir}...")
        
        proteins_list = []
        compounds_list = []
        adjacencies_list = []
        regression_list = []

        Kcat_data = pd.read_csv(data_path)

        for index, row in tqdm(Kcat_data.iterrows(), total=Kcat_data.shape[0]):
            smiles = row['Smiles']
            sequence = row['Sequence']
            Kcat = row['Value']

            if "." not in smiles and float(Kcat) > 0:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                if mol is None: # Handle invalid SMILES
                    tqdm.write(f"Warning: Invalid SMILES skipped for transformation: {smiles}")
                    continue

                atoms = self.create_atoms(mol)
                i_jbond_dict = self.create_ijbonddict(mol)
                fingerprints = self.extract_fingerprints(atoms, i_jbond_dict)
                compounds_list.append(fingerprints)

                adjacency = self.create_adjacency(mol)
                adjacencies_list.append(adjacency)

                words = self.split_sequence(sequence)
                proteins_list.append(words)

                regression_list.append(np.array([math.log2(float(Kcat))]))
            else:
                tqdm.write(f"Skipping entry (invalid SMILES or Kcat<=0) for transformation: SMILES={smiles}, Kcat={Kcat}")
        
        # Save processed data
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def tensor_long(array_data):
            # Ensure elements are not empty lists/arrays
            filtered_data = [sublist for sublist in array_data if len(sublist) > 0]
            if not filtered_data:
                return [] # Return empty list if no valid data
            return [torch.tensor(sublist, dtype=torch.long) for sublist in filtered_data]

        def tensor_float(array_data):
            filtered_data = [sublist for sublist in array_data if len(sublist) > 0]
            if not filtered_data:
                return []
            return [torch.tensor(sublist, dtype=torch.float) for sublist in filtered_data]

        with open(os.path.join(output_dir, 'compounds.pkl'), 'wb') as f:
            pickle.dump(tensor_long(compounds_list), f)
        with open(os.path.join(output_dir, 'adjacencies.pkl'), 'wb') as f:
            pickle.dump(tensor_float(adjacencies_list), f)
        with open(os.path.join(output_dir, 'regression.pkl'), 'wb') as f:
            pickle.dump(tensor_float(regression_list), f)
        with open(os.path.join(output_dir, 'proteins.pkl'), 'wb') as f:
            pickle.dump(tensor_long(proteins_list), f)
        
        print(f"Transformation complete for {data_path}. Data saved to {output_dir}")


def main_workflow():
    # Define paths
    train_data_path = '../../Data/Train_kcat_prottrans_log2.csv' # Assuming you have a train CSV
    test_data_path = '../../Data/Test_kcat_prottrans_log2.csv'

    # Define output directories
    output_base_dir = '../../Data/xw_input_kkm/' # Base directory for all outputs
    dictionaries_dir = os.path.join(output_base_dir, 'dictionaries')
    train_output_dir = os.path.join(output_base_dir, 'train_data')
    test_output_dir = os.path.join(output_base_dir, 'test_data')

    # Create directories if they don't exist
    os.makedirs(dictionaries_dir, exist_ok=True)
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Step 1: Initialize preprocessor and build dictionaries from training data
    preprocessor = DataPreprocessor(radius=2, ngram=3)
    preprocessor.fit_dictionaries(train_data_path)
    preprocessor.dump_dictionaries(dictionaries_dir)
    print(f"Dictionaries saved to {dictionaries_dir}")

    # Step 2: Process training data using the built dictionaries
    # (Optional: If you want to rebuild dicts based on train only, and save, then reload for test.
    # But usually, fit_dictionaries already populates the preprocessor's dicts.
    # We can just use the current state of preprocessor for train data generation too.)
    # If you closed the script and restarted, you'd load dicts before step 2/3.
    # preprocessor.load_dictionaries(dictionaries_dir) # Only needed if restarting script

    preprocessor.transform_data(train_data_path, train_output_dir)

    # Step 3: Process test data using the *same* dictionaries
    # (The preprocessor instance still holds the dictionaries built from train data)
    preprocessor.transform_data(test_data_path, test_output_dir)

if __name__ == '__main__':
    main_workflow()