import os
import sys

import pandas as pd
from rdkit import RDLogger

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from dfpl import fingerprint as fp

correct_smiles = [
    "CC1(C)OC2CC3C4CC(F)C5=CC(=O)CCC5(C)C4C(O)CC3(C)C2(O1)C(=O)CO",
    "CC1(C)OC2CC3C4CCC5=CC(=O)C=CC5(C)C4(F)C(O)CC3(C)C2(O1)C(=O)CO",
    "CC12CCC(=O)C=C1CCC3C2CCC4(C)C3CCC4(O)C#C",
    "CC1CC2C3CC(F)C4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
    "CC1CC2C3CC(F)C4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(OC(=O)C)C(=O)COC(=O)C",
    "CC1CC2C3CCC(O)(C(=O)CO)C3(C)CC(O)C2C4(C)C=CC(=O)C=C14",
    "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
    "CCC(=O)OC1(C(C)CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC12C)C(=O)CCl",
]

incorrect_smiles = [
    "CCN(CC)c1ccc(C(O)c2ccccc2C(O)O)c(O)c3",
    "CC[N+](CC)(CCCCOc1ccc(C=Cc2ccccc2)cc1",
    "O=COc1ccccc)c2cccc(c2=O)Oc3ccccc3",
    "SCC3(O)OCOCS",
    "C(CSc1ccccc1Sc2ccccc2",
    "CC(=O)OCC(=O1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C",
    "CC12CCC3C(CCC4)C(=O)CCC34C)C1CCC2=O",
    "CCn1c2ccccc2c3ccc13)[N+](=O)[O-]",
    "COC(=Occc2[nH]1)S(=O)c3ccccc3",
    "Cc1ccc(Oc1)n2nc3ccccc3n2",
    "NNc1cccc1)C(=O)O",
    "Nc1c2CCCnc3ccccc13",
    "Nc1ccccccc2c1",
    "Occc(cc1)C(=O)Oc2ccccc2",
    "[O][N+](=N1ccccc1)c2ccccc2",
    "(cc1)c2ccc(cc2)c3ccccc3",
]


def test_correct_mol_examples():
    df = pd.DataFrame(correct_smiles, columns=["smiles"])
    df = fp.addFPColumn(df, fp_size=2048)
    allNotNone = df[df["fp"].notnull()]
    assert len(allNotNone.index) == len(correct_smiles)


def test_incorrect_smiles():
    RDLogger.DisableLog("rdApp.*")
    df = pd.DataFrame(incorrect_smiles, columns=["smiles"])
    df = fp.addFPColumn(df, fp_size=2048)
    allNotNone = df[df["fp"].notnull()]
    assert len(allNotNone.index) == 0
