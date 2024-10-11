import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt

# Define common functional groups using SMARTS
functional_groups = {
    'Alkyl Halide': '[CX4][Cl,Br,I,F]',
    'Alcohol': '[CX4][OH]',
    'Carboxylic Acid': 'C(=O)[OH]',
    'Ester': '[CX3](=O)[OX2H0]',
    'Amine': '[NX3;H2,H1,H0;!$(NC=O)]',
    'Amide': '[NX3][CX3](=[OX1])[#6]',
    'Nitro': '[NX3](=O)[O-]',
    'Alkyne': '[C]#C',
    'Aromatic Ring': 'a1aaaaa1'
}


# Function to fragment a molecule using BRICS
def fragment_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    fragments = BRICS.BRICSDecompose(mol)
    return list(fragments)


# Function to calculate properties for a fragment
def calculate_fragment_properties(fragment_smiles):
    mol = Chem.MolFromSmiles(fragment_smiles)
    if not mol:
        return None
    properties = {
        'Molecular Weight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'Num H Donors': Descriptors.NumHDonors(mol),
        'Num H Acceptors': Descriptors.NumHAcceptors(mol)
    }
    return properties


# Function to check if a fragment contains a specific functional group
def check_functional_group(fragment_smiles, functional_groups):
    mol = Chem.MolFromSmiles(fragment_smiles)
    for group_name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            return group_name
    return 'Other'


# Load the dataset (assumed to have 'smiles', 'AR', and 'predicted' columns)
data = pd.read_csv('predict_data_AR.csv')

# Prepare a DataFrame to store fragment-based properties for each molecule
fragments_data = []

# Loop through each molecule in the dataset
for idx, row in data.iterrows():
    smiles = row['smiles']
    fragments = fragment_molecule(smiles)  # Fragment the molecule

    if fragments:
        for frag in fragments:
            frag_props = calculate_fragment_properties(frag)  # Calculate fragment properties
            functional_group = check_functional_group(frag, functional_groups)  # Get functional group

            # Append fragment data to the results
            fragments_data.append({
                'SMILES': smiles,
                'Fragment': frag,
                'AR': row['AR'],
                'Predicted': row['predicted'],
                'Functional Group': functional_group,
                'Molecular Weight': frag_props['Molecular Weight'],
                'LogP': frag_props['LogP'],
                'Num H Donors': frag_props['Num H Donors'],
                'Num H Acceptors': frag_props['Num H Acceptors']
            })

# Convert the fragments data to a DataFrame for analysis
fragments_df = pd.DataFrame(fragments_data)
fragments_df.to_csv('fragments.csv', index=False)

# Display the first few rows of the fragment-based data
print(fragments_df.head())

# Optional: Save the results to a CSV file
fragments_df.to_csv('fragment_analysis_results.csv', index=False)

# Example: Visualizing the distribution of functional groups across all fragments
plt.figure(figsize=(10, 10))
fragments_df['Functional Group'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Functional Groups in Fragments')
plt.xlabel('Functional Group')
plt.ylabel('Count')
plt.show()
