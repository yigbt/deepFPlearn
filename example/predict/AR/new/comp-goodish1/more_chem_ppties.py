import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, rdPartialCharges
from rdkit.Chem import rdFreeSASA
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem

# Load your data
data = pd.read_csv('predict_data_AR.csv')

# Define functions to calculate molecular properties
def calculate_partial_charges(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
        return sum(charges)  # Total charge (or return list of atom charges if desired)
    else:
        return None

def calculate_rotatable_bonds(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    return rdMolDescriptors.CalcNumRotatableBonds(mol) if mol else None


def calculate_lipinski(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        # Checking Lipinski's rule of five violations
        mw = Descriptors.MolWt(mol)  # Molecular weight
        logp = Crippen.MolLogP(mol)  # LogP
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        violations = int(mw > 500) + int(logp > 5) + int(h_donors > 5) + int(h_acceptors > 10) + int(rotatable_bonds > 10)
        return violations  # Number of rule violations
    return None

def calculate_surface_area(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        mol = Chem.AddHs(mol)
        res = Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDG())
        if (res < 0):
            print(f"Failed to embed molecule {smiles_string}")
            return None
        radii = rdFreeSASA.classifyAtoms(mol)
        return rdFreeSASA.CalcSASA(mol, radii)
    return None




# Toxicophore analysis (use a curated list of toxicophores, such as ToxTree/Derek Nexus)
# This requires external models/tools. Example below assumes pre-trained model or database:

# Apply the calculation functions to your dataset
data['partial_charge'] = data['smiles'].apply(calculate_partial_charges)
data['rotatable_bonds'] = data['smiles'].apply(calculate_rotatable_bonds)

data['lipinski_violations'] = data['smiles'].apply(calculate_lipinski)
data['molecular_surface_area'] = data['smiles'].apply(calculate_surface_area)



properties = ['partial_charge', 'rotatable_bonds',  'lipinski_violations', 'molecular_surface_area']
titles =['Partial charge', 'Rotatable bonds', 'Lipinski', 'Molecular Surface Area']

data = data.dropna(subset=['partial_charge', 'rotatable_bonds',  'lipinski_violations', 'molecular_surface_area' ])

fig, axes = plt.subplots(2, 2, figsize=(17, 14))

# for ax, prop, title in zip(axes.flatten(), properties, titles):
#     scatter = ax.scatter(
#         data['AR'], data['predicted'],
#         c=data[prop], cmap='viridis', s=20
#     )
#     # Add color bar to each subplot
#     cbar = plt.colorbar(scatter, ax=ax)
#     cbar.set_label(title)
#     ax.plot([0, 250], [0, 250], '--', color='red', label='45-degree line')
#     ax.set_xlim([0,250])
#     ax.set_ylim([0,250])
#     # Set labels and title for each subplot
#     ax.set_title(f'Actual vs Predicted with {title}')
#     ax.set_xlabel('Actual')
#     ax.set_ylabel('Predicted')
for prop in properties:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[prop], bins=30)
    plt.title(f'Distribution of {prop}')
    plt.xlabel(prop)
    plt.ylabel('Frequency')
    plt.show()

#plt.tight_layout()
#plt.show()