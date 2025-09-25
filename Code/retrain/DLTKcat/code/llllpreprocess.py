import pandas as pd

from rdkit import Chem
from rdkit.Chem import MolStandardize

class MolClean(object):
    def __init__(self):
        self.normizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()
 
    def clean(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = self.normizer.normalize(mol)
            mol = self.lfc.choose(mol)
            mol = self.uc.uncharge(mol)
            smi = Chem.MolToSmiles(mol,  isomericSmiles=False, canonical=True)
            return smi
        else:
            return None
organism = 'human'
df = pd.read_csv(f'/home/wuke/project/bio_deeplearning/kcatkm_predict/DLTKcat/data/{organism}_kcat_pre_pd.csv')

# 1    not isometric
mol_cleaner = MolClean()
df['smiles'] = df['Smiles'].apply(mol_cleaner.clean)


# 2    is isometric
# df = df.rename(columns={'Smiles':'smiles'})
# df = df[~df['smiles'].str.contains(r'\\')]

df = df.rename(columns={'Sequence':'seq'})
df['Temp_K_norm'] = 0.3
df['Inv_Temp_norm'] = 0.6307273626917579
df.to_csv('../data/changed_data.csv', index=None)
# print(df)
# gdx = df['smiles'].head(64).unique()
# print(gdx)

# rows_per_file = 10

# # 计算总共需要的文件数量
# num_files = len(df) // rows_per_file + (1 if len(df) % rows_per_file != 0 else 0)

# for i in range(num_files):
#     start_row = i * rows_per_file
#     end_row = (i + 1) * rows_per_file
#     df_subset = df.iloc[start_row:end_row]
#     output_filename = f'../data/input_{i+1}.csv'
#     df_subset.to_csv(output_filename, index=False)
#     print(f'File {output_filename} written with rows from {start_row} to {end_row-1}')

