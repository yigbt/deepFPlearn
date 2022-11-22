import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem

def inchi2smiles(inchi):
    try:
        return np.array(
            Chem.MolToSmiles(Chem.MolFromInchi(inchi)))
    except:
        # Note: We don't need to log here since rdkit already logs
        return None
df = pd.read_table('/home/soulios/git-soulios/DEEPFPLEARN/dfpl_DBN/data/D_dataset.tsv',sep ='\t',header=3)
first_column = df.iloc[:, 1]
smiles = []
for i in first_column:
    smiles.append(inchi2smiles(i))
smilesdf = pd.DataFrame(smiles)
smilesdf.to_csv('D_dataset_smiles.csv')