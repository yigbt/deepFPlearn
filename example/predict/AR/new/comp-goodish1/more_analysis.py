import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen

# Calculate residuals (error) between predicted and actual values
data = pd.read_csv('predict_data_AR.csv')
data = data.dropna()
data['residuals'] = data['predicted'] - data['AR']

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

def calculate_rotatable_bonds(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    return rdMolDescriptors.CalcNumRotatableBonds(mol) if mol else None




# 1. Scatter Plot: Predicted vs Actual (with error visualization)
plt.figure(figsize=(8, 6))
# Store seaborn scatter plot in a variable
sc = sns.scatterplot(data=data, x='AR', y='predicted', hue='residuals', palette='coolwarm', s=100)
plt.title('Predicted vs Actual Toxicity (Colored by Residuals)')
plt.xlabel('Actual Toxicity (AR)')
plt.ylabel('Predicted Toxicity')
plt.colorbar(sc.collections[0], label='Residuals')
plt.show()

# 2. Residuals Plot: Residuals vs Actual
plt.figure(figsize=(8, 6))
# Store seaborn scatterplot in a variable
sns.scatterplot(data=data, x='AR', y='residuals')
plt.title('Residuals vs Actual Toxicity ')
plt.xlabel('Actual Toxicity (AR)')
plt.ylabel('Residuals (Predicted - Actual)')

# Please note the modification here

plt.show()




# 3. Correlation and R²
correlation = np.corrcoef(data['AR'], data['predicted'])[0, 1]
r2 = r2_score(data['AR'], data['predicted'])

print(f"Correlation between predicted and actual: {correlation:.3f}")
print(f"R² score: {r2:.3f}")

# 4. Error Distribution (Histogram)
plt.figure(figsize=(8, 6))
sns.histplot(data['residuals'], bins=20, kde=True)
plt.title('Distribution of Residuals (Prediction Errors)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--')
plt.show()

data['tpsa'] = data['smiles'].apply(calculate_tpsa)
data['molecular_weight'] = data['smiles'].apply(calculate_molecular_weight)
data['logp'] = data['smiles'].apply(calculate_logp)
data['rotatable_bonds'] = data['smiles'].apply(calculate_rotatable_bonds)



# 5. Analyze Residuals with Molecular Properties (e.g., LogP)
properties = ['logp', 'molecular_weight', 'tpsa', 'rotatable_bonds']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, prop in zip(axes.flatten(), properties):
    sns.scatterplot(data=data, x=prop, y='residuals', ax=ax, palette='coolwarm', hue='residuals', s=100)
    ax.set_title(f'Residuals vs {prop.capitalize()}')
    ax.axhline(0, color='red', linestyle='--')

plt.tight_layout()
plt.show()
