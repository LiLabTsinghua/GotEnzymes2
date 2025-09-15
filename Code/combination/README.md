# Combination repository

This repository provides code for predicting protein-molecule interactions using pretrained protein and molecular language models, followed by ensemble learning with multiple models.

## 📁 Project Structure

```
.
├── DLKcat_EITLEM_CataPro/      # Contains three ensemble model implementations
├── ExtraTrees.py               # Alternative ensemble model using Extra Trees
└── README.md
```

## 🚀 Workflow Overview

1. **Generate Embeddings**  
   Use pretrained protein and molecular language models to extract embeddings for your sequences and compounds.

2. **Save Embeddings**  
   Save protein embeddings as a `.pkl` file using a dictionary format:  
   ```python
   {sequence: embedding}  # e.g., {"MKTVR...": [0.1, -0.5, ...]}
   ```
   > ⚠️ Note: Some models require **token-level** embeddings, while others use **sequence-level** (e.g., [CLS] token or mean pooling). Make sure to match the required format. Embeddings can take significant disk space.

3. **Combine Protein-Molecule Pairs**  
   Pair protein and molecule embeddings for downstream prediction.

4. **Train & Evaluate Models**  
   You have **four** options for the final prediction model:
   - Use **any of the three models** implemented in the `DLKcat_EITLEM_CataPro` folder.
   - Use the **`ExtraTrees.py`** script as an alternative ensemble method (e.g., Extra-Trees classifier/regressor).

## 🧬 Pretrained Models Used

### Protein Language Models
| Model       | Repository |
|------------|------------|
| ESM-1b, ESM-1v, ESM2, ESM-C | [evolutionaryscale/esm](https://github.com/evolutionaryscale/esm) |
| ProtT5     | [agemagician/ProtTrans](https://github.com/agemagician/ProtTrans) |
| ProLLaMA   | [PKU-YuanGroup/ProLLaMA](https://github.com/PKU-YuanGroup/ProLLaMA) |

### Molecular Models
| Model             | Repository |
|------------------|-----------|
| Mole-BERT         | [junxia97/Mole-BERT](https://github.com/junxia97/Mole-BERT) |
| ChemBERTa-2       | [miservilla/ChemBERTa](https://github.com/miservilla/ChemBERTa) |
| UniMol V1, V2     | [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol) |
| MolGen            | [zjunlp/MolGen](https://github.com/zjunlp/MolGen) |
| SMILES Transformer| [DSPsleeporg/smiles-transformer](https://github.com/DSPsleeporg/smiles-transformer) |
| Molecular Fingerprint | [rdkit/rdkit](https://github.com/rdkit/rdkit) |

## 🛠 How to Run

### Step 1: Generate and Save Embeddings
Example:
```python
import pickle

# After obtaining embeddings
protein_embeddings = {"ACDEFGHIKLMNPQRST": embedding_vector}
with open("esm2.pkl", "wb") as f:
    pickle.dump(protein_embeddings, f)
```

### Step 2: Use Ensemble Models
- To use one of the **three models** in `DLKcat_EITLEM_CataPro/`, navigate into the folder and follow its instructions.
- Alternatively, run:
  ```bash
  python ExtraTrees.py
  ```
  Make sure input embeddings are properly loaded and paired.

## 📚 Notes
- Ensure consistent embedding dimensions and pooling strategies across models.
- The choice between token-level and sequence-level embeddings depends on the downstream model design.
