# Load necessary libraries
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import rdchem
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import Lipinski

data = pd.read_csv('predict_data_AR.csv')

def calculate_molecular_weight(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            return Descriptors.MolWt(mol)
        else:
            return None
    except:
        return None

def calculate_logp(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            return Crippen.MolLogP(mol)
        else:
            return None
    except:
        return None

def calculate_tpsa(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            return rdMolDescriptors.CalcTPSA(mol)
        else:
            return None
    except:
        return None

def calculate_num_h_donors(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            return Lipinski.NumHDonors(mol)
        else:
            return None
    except:
        return None
def calculate_num_rotatable_bonds(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            return rdMolDescriptors.CalcNumRotatableBonds(mol)
        else:
            return None
    except:
        return None

def calculate_ring_count(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            return mol.GetRingInfo().NumRings()
        else:
            return None
    except:
        return None

def calculate_molecular_complexity(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            return GraphDescriptors.BertzCT(mol)
        else:
            return None
    except:
        return None

def calculate_num_h_acceptors(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol:
            return Lipinski.NumHAcceptors(mol)
        else:
            return None
    except:
        return None


data['molecular_weight'] = data['smiles'].apply(calculate_molecular_weight)
data['logp'] = data['smiles'].apply(calculate_logp)
data['tpsa'] = data['smiles'].apply(calculate_tpsa)
data['num_h_donors'] = data['smiles'].apply(calculate_num_h_donors)
data['num_rotatable_bonds'] = data['smiles'].apply(calculate_num_rotatable_bonds)
data['ring_count'] = data['smiles'].apply(calculate_ring_count)
data['molecular_complexity'] = data['smiles'].apply(calculate_molecular_complexity)
data['num_h_acceptors'] = data['smiles'].apply(calculate_num_h_acceptors)

print(data['molecular_weight'][0])
print(data[['molecular_weight', 'logp',  'num_h_donors',  'ring_count', 'molecular_complexity', 'num_h_acceptors']].isna().sum())


 #Drop rows with invalid SMILES (optional)
data = data.dropna(subset=['molecular_weight', 'logp',  'num_h_donors', 'ring_count', 'molecular_complexity', 'num_h_acceptors','tpsa','num_rotatable_bonds'])


fig, axes = plt.subplots(4, 2, figsize=(17, 14))

properties = ['molecular_weight', 'logp',  'num_h_donors',  'ring_count','molecular_complexity' , 'num_h_acceptors','tpsa','num_rotatable_bonds']
titles =  ['Molecular Weight', 'LogP' ,  'Num H Donors' ,  'Ring count', 'Molecular Complexity', 'Num H Acceptors','TPSA','Num Rotatable Bonds']

#for i in properties:
 #   print(data[str(i)])


for ax, prop, title in zip(axes.flatten(), properties, titles):
    scatter = ax.scatter(
        data['AR'], data['predicted'],
        c=data[prop], cmap='viridis', s=20
    )
    # Add color bar to each subplot
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(title)
    ax.plot([0, 250], [0, 250], '--', color='red', label='45-degree line')
    ax.set_xlim([0,250])
    ax.set_ylim([0,250])
    # Set labels and title for each subplot
    ax.set_title(f'Actual vs Predicted with {title}')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

#plt.tight_layout()
plt.show()
