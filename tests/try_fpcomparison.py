import dfpl.fingerprint as fp
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw

import pandas as pd
import numpy as np

from dfpl import autoencoder as ac
from dfpl import feedforwardNN as fNN
from dfpl import predictions
from dfpl import options as opt

# read both datasets
dfS = fp.importDataFile("data/S_dataset_extended.pkl", import_function=fp.importSmilesCSV, fp_size=2048)
dfS.dropna(axis=0, subset=['cid'], inplace=True)
dfS['cid'] = dfS['cid'].apply(int).astype(str)
dfD = fp.importDataFile("data/dsstox_20160701.pkl", import_function=fp.importSmilesCSV, fp_size=2048)

# ids and structures of interest
cid_of_interest = ["87587", "77328", "2734118", "2736548", "154257"]
toxid_of_interest = ["DTXSID3027798", "DTXSID7041461", "DTXSID9048067", "DTXSID7049344", "DTXSID70173593"]
df = pd.DataFrame(list(zip(cid_of_interest, toxid_of_interest)), columns=["cid", "toxid"])

# add smiles
smiles_of_interest = dfS[dfS['cid'].isin(cid_of_interest)][['cid', 'smiles']]
df = df.merge(smiles_of_interest, on="cid")

# add inchi
inchi_of_interest = dfD[dfD['toxid'].isin(toxid_of_interest)][['toxid', 'inchi']]
df = df.merge(inchi_of_interest, on="toxid")

# get pre-calculated boolean fingerprints from input .pkl data
fpboolS_of_interest = dfS[dfS['cid'].isin(cid_of_interest)][['cid', 'fp']]
df = df.merge(fpboolS_of_interest, on="cid")
df.rename(columns={'fp': 'fpSbool'}, inplace=True)
fpboolD_of_interest = dfD[dfD['toxid'].isin(toxid_of_interest)][['toxid', 'fp']]
df = df.merge(fpboolD_of_interest, on="toxid")
df.rename(columns={'fp': 'fpDbool'}, inplace=True)

# calculate AND result of bool fingerprints
df['allBoolEqual'] = [all(s == d) for s, d in zip(df['fpSbool'].to_list(), df['fpDbool'].to_list())]

# generate binary fingerprints
df['fpSbin'] = [Chem.RDKFingerprint(Chem.MolFromSmiles(x)) for x in df['smiles']]
df['fpDbin'] = [Chem.RDKFingerprint(Chem.MolFromInchi(x)) for x in df['inchi']]

# calculate Tanimoto Similarity of both compounds
df['tanimoto'] = [DataStructs.FingerprintSimilarity(s, d) for s, d in zip(df['fpSbin'], df['fpDbin'])]

# generate mol structures for drawing
df['molS'] = [Chem.MolFromSmiles(x) for x in df['smiles']]
df['molD'] = [Chem.MolFromInchi(x) for x in df['inchi']]

legend = [c + " (" + str(round(t, 2)) + ", bool: " + str(b) + ")" for c, t, b in
          zip(df['cid'], df['tanimoto'], df['allBoolEqual'])] + \
         [t for t in df['toxid']]

img = Draw.MolsToGridImage(df['molS'].to_list() + df['molD'].to_list(),
                           molsPerRow=df.shape[0],
                           subImgSize=(200, 200),
                           legends=legend)
img.save('cidVStoxid.structures.png')
img.show()

project_directory = ""
opts = opt.PredictOptions(
    inputFile=f"",
    outputDir=f"/home/hertelj/tmp/",
    model=f"/home/hertelj/git-hertelj/deepFPlearn_CODE/validation/case_03/results/ER_compressed-True_sampled-None.best.FNN.model.hdf5",
    target="ER",
    fpSize=2048,
    type="smiles",
    fpType="topological"
)

(_, encoder) = ac.define_ac_model(input_size=2048, encoding_dim=256)
encoder.load_weights("/home/hertelj/git-hertelj/deepFPlearn_CODE/modeltraining/Sun_etal_dataset.encoder.hdf5")
data = ac.compress_fingerprints(dfS, encoder)
s_compressed = data[data['cid'].isin(cid_of_interest)]['fpcompressed']

df2 = predictions.predict_values(df=data,
                                 opts=opts,
                                 use_compressed=True)
s_predictions = df2[df2['cid'].isin(cid_of_interest)][['cid', 'trained']]

data2 = ac.compress_fingerprints(dfD, encoder)
d_compressed = data2[data2['toxid'].isin(toxid_of_interest)]['fpcompressed']
df3 = predictions.predict_values(df=data2,
                                 opts=opts,
                                 use_compressed=True)
d_predictions = df3[df3['toxid'].isin(toxid_of_interest)][['toxid', 'trained']]

fp_matrix_S = np.array(df['fpSbool'].to_list(), dtype=bool, copy=False)
predictions_S = encoder.predict(fp_matrix_S)
fp_matrix_D = np.array(df['fpDbool'].to_list(), dtype=bool, copy=False)
predictions_D = encoder.predict(fp_matrix_D)
df['fpcompressedS'] = [s for s in predictions_S]
df['fpcompressedD'] = [s for s in predictions_D]

# compressed fp equal?
df['fpcEqual'] = [all(s == d) for s, d in zip(df['fpcompressedS'].to_list(), df['fpcompressedD'].to_list())]
