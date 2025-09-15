# You can find the codes of pretrained language models from these GitHub repositories:

### Protein language models:
- ESM-1b, ESM-1v, ESM2, ESM C: https://github.com/evolutionaryscale/esm
- ProtT5: https://github.com/agemagician/ProtTrans
- ProLLaMA: https://github.com/PKU-YuanGroup/ProLLaMA
### Protein language models:
- Mole-BERT: https://github.com/junxia97/Mole-BERT
- ChemBERTa-2: https://github.com/miservilla/ChemBERTa
- UniMol V1, V2: https://github.com/deepmodeling/Uni-Mol
- MolGen: https://github.com/zjunlp/MolGen
- SMILES Transformer: https://github.com/DSPsleeporg/smiles-transformer

After you get the embedding of your proteins using a pretrained language model, you could should them as a pickle file.

Here, we use a dictionary whose key is sequence itself, and value is its embedding.

Attention: Some of the models need a token-level embedding, while others need a sequence-level embedding. You may need a large space to save them.