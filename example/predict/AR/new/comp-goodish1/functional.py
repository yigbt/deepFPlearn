import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

# Define common functional groups using SMARTS
functional_groups = {
    'Alkyl Halide': '[CX4][Cl,Br,I,F]',  # Alkyl halides
    'Alcohol': '[CX4][OH]',  # Alcohols
    'Carboxylic Acid': 'C(=O)[OH]',  # Carboxylic acids
    'Ester': '[CX3](=O)[OX2H0]',  # Esters
    'Amine': '[NX3;H2,H1,H0;!$(NC=O)]',  # Amines
    'Amide': '[NX3][CX3](=[OX1])[#6]',  # Amides
    'Nitro': '[NX3](=O)[O-]',  # Nitro groups
    'Alkyne': '[C]#C',  # Alkynes
    'Aromatic Ring': 'a1aaaaa1'  # Aromatic rings (benzene-like)
}

# Function to assign functional group based on SMARTS matching
def assign_functional_group(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if not mol:
        return None
    for group_name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            return group_name
    return 'Other'  # If no functional group is matched

# Load the dataset
data = pd.read_csv('predict_data_AR.csv')

# Assign functional groups to each molecule
data['Functional Group'] = data['smiles'].apply(assign_functional_group)

# Drop rows where the SMILES string is invalid or functional group is not identified
data = data.dropna(subset=['Functional Group'])

# Create a scatter plot of predicted vs AR, with color based on functional group
plt.figure(figsize=(10, 6))
groups = data.groupby('Functional Group')

# Define a color map for the functional groups
colors = plt.cm.get_cmap('tab10', len(groups))

# Scatter plot with colors based on functional groups
for i, (name, group) in enumerate(groups):
    plt.scatter(group['AR'], group['predicted'], label=name, alpha=0.7, s=100, cmap=colors(i))

# Labels and title
plt.xlabel('AR')
plt.ylabel('Predicted')
plt.title('Predicted vs AR Colored by Functional Group')
plt.legend(title="Functional Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
