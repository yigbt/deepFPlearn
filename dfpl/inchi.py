import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
import requests
import csv
import time
from multiprocessing import Pool
# def inchi2smiles(inchi):
#     try:
#         return np.array(
#             Chem.MolToSmiles(Chem.MolFromInchi(inchi)))
#     except:
#         # Note: We don't need to log here since rdkit already logs
#         return None
# df = pd.read_table('~/deepFPlearn/example/data/D_dataset.tsv',sep ='\t',header=3)
# print(len(df))
# first_column = df.iloc[:, 1]
# smiles = []
# for i in first_column:
#     smiles.append(inchi2smiles(i))
# smilesdf = pd.DataFrame(smiles)
# print(len(smilesdf))
# smilesdf.to_csv('D_dat_smiles.csv')


# # Read InChIKeys from input TSV file
# in_file = 'example/data/D_fragment.csv'
# in_keys = []
# with open(in_file, 'r') as f:
#     reader = csv.DictReader(f, delimiter=',')
#     for row in reader:
#         in_keys.append(row['inchikey'])
#
# # Define base URL for PubChem RESTful API
# base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
#
# # Define function to retrieve canonical SMILES for a given InChIKey
#
#
# def get_smiles(inchikey):
#     url = f"{base_url}/compound/inchikey/{inchikey}/property/CanonicalSMILES/txt"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.text.strip()
#     else:
#         return None
#
#
# # Retrieve canonical SMILES for all InChIKeys and write to output CSV file
# out_file = 'smiles_D.csv'
# num_retrieved = 0
# with open(out_file, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['inchikey', 'canonical_smiles'])
#     for inchikey in in_keys:
#         smiles = get_smiles(inchikey)
#         if smiles is not None:
#             num_retrieved += 1
#             writer.writerow([inchikey, smiles])
#
# # Write number of retrieved molecules to output CSV file
# with open(out_file, 'a', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(
#         ['', f'Retrieved {num_retrieved} molecules out of {len(in_keys)}'])

# Read InChIKeys from input TSV file
# in_file = 'example/data/D_fragment.csv'
# in_keys = []
# with open(in_file, 'r') as f:
#     reader = csv.DictReader(f, delimiter=',')
#     for row in reader:
#         in_keys.append(row['inchikey'])
#
# # Define base URL for PubChem RESTful API
# base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
#
# # Define function to retrieve first canonical SMILES for a given InChIKey
# def get_first_smiles(inchikey):
#     url = f"{base_url}/compound/inchikey/{inchikey}/property/CanonicalSMILES/txt"
#     with requests.Session() as session:
#         response = session.get(url)
#         if response.status_code == 200:
#             smiles_list = response.text.strip().split('\n')
#             return smiles_list[0]
#         else:
#             return None
#
# # Define function to retrieve canonical SMILES for a batch of InChIKeys using multiprocessing
# def retrieve_smiles_batch(in_keys):
#     with Pool() as pool:
#         smiles_list = pool.map(get_first_smiles, in_keys)
#     return smiles_list
#
# # Retrieve canonical SMILES for all InChIKeys using multiprocessing and print number of retrieved molecules and elapsed time
# start_time = time.time()
# smiles_list = retrieve_smiles_batch(in_keys)
# num_retrieved = sum([1 for smiles in smiles_list if smiles is not None])
# elapsed_time = time.time() - start_time
# print(f"Retrieved {num_retrieved} molecules out of {len(in_keys)} in {elapsed_time:.2f} seconds")
#
# # Write number of retrieved molecules to output CSV file
# out_file = 'output_file.csv'
# with open(out_file, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['inchikey', 'canonical_smiles'])
#     for inchikey, smiles in zip(in_keys, smiles_list):
#         if smiles is not None:
#             writer.writerow([inchikey, smiles])
#
# # Print final message with elapsed time
# print(f"Retrieved {num_retrieved} molecules out of {len(in_keys)} in {elapsed_time:.2f} seconds")
import requests
import csv

# Read InChIKeys from input TSV file
in_file = 'example/data/D_fragment.csv'
in_keys = []
with open(in_file, 'r') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        in_keys.append(row['inchikey'])

# Define base URL for PubChem RESTful API
base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'

# Define function to retrieve first canonical SMILES for a given InChIKey from a list of canonical SMILES
def get_first_smiles(smiles_list):
    if len(smiles_list) > 0:
        return smiles_list[0]
    else:
        return None

# Define function to retrieve canonical SMILES for a list of InChIKeys using bulk download
def retrieve_smiles_bulk(in_keys, batch_size=100):
    smiles_dict = {}
    for i in range(0, len(in_keys), batch_size):
        sub_keys = in_keys[i:i+batch_size]
        id_list = ','.join([f'InChIKey={inchikey}' for inchikey in sub_keys])
        url = f"{base_url}/compound/property/CanonicalSMILES/json?{id_list}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for result in data['PropertyTable']['Properties']:
                try:
                    inchikey = result['CID']['InChIKey']
                    smiles_list = result['CanonicalSMILES']
                    smiles = get_first_smiles(smiles_list)
                    if smiles is not None:
                        smiles_dict[inchikey] = smiles
                except Exception as e:
                    print(f"Error retrieving data for InChIKey {inchikey}. Exception: {e}")
        else:
            print(f"Error retrieving data for InChIKeys {sub_keys}. Response code: {response.status_code}")
    return smiles_dict

# Retrieve first canonical SMILES for all InChIKeys using bulk download and print number of retrieved molecules
smiles_dict = retrieve_smiles_bulk(in_keys)
num_retrieved = len(smiles_dict)
print(f"Retrieved {num_retrieved} molecules out of {len(in_keys)}")

# Write number of retrieved molecules to output CSV file
out_file = 'output_file.csv'
with open(out_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['inchikey', 'canonical_smiles'])
    for inchikey, smiles in smiles_dict.items():
        writer.writerow([inchikey, smiles])