# Load necessary libraries

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.Chem import Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('predict_data_AR.csv')





# def calcTPSA(smiles_string):
#     try:
#         mol = Chem.MolFromSmiles(smiles_string)
#         if mol:
#             return rdMolDescriptors.CalcTPSA(mol)
#         else:
#             return None
#     except:
#         return None
#
#
# data['calcTPSA'] = data['smiles'].apply(calcTPSA)
#
# print(data)

def num_radical_electrons(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            return rdMolDescriptors.CalcNumHeavyAtoms(mol)
        else:
            return None
    except:
        return None
data['num_radical_electrons'] = data['smiles'].apply(num_radical_electrons)
print(data)

 #Drop rows with invalid SMILES (optional)
# data = data.dropna(subset=['smiles','calcTPSA'])
#
#
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(data['AR'], data['predicted'], c=data['calcTPSA'], cmap='viridis', s=20, alpha=0.7)
# plt.colorbar(scatter, label='TPSA')
# plt.xlim([-5,3])
# plt.ylim([-5,3])
# plt.plot([-5, 3], [-5, 3], 'r--', linewidth=1)
#
# plt.xlabel('AR')
# plt.ylabel('Predicted')
# plt.title('Predicted vs AR with TPSA Color Gradient')
# plt.grid(True)
# plt.show()


data = data.dropna(subset=['smiles','num_radical_electrons'])
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['AR'], data['predicted'], c=data['num_radical_electrons'], cmap='viridis', s=20, alpha=0.7)
plt.colorbar(scatter, label='radical electrons')
plt.xlim([-5,3])
plt.ylim([-5,3])
plt.plot([-5, 3], [-5, 3], 'r--', linewidth=1)
plt.xlabel('AR')
plt.ylabel('Predicted')
plt.title('Predicted vs actual with num of radical electrons Color Gradient')
plt.grid(True)
plt.show()

#for i in properties:
 #   print(data[str(i)])


