from __future__ import annotations

import logging
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from rdkit import Chem, DataStructs
from simple_parsing import ArgumentParser, choice
from simple_parsing.helpers import Serializable

# Use this conda env: mamba create - n tidyChem - c conda - forge r - base r - tidyverse r - stringdist python rdkit
# numpy pandas simple - parsing


@dataclass
class ProgramConfig(Serializable):
    """
    Configuration options for running the program
    """

    # file containing SMILES or InChikeys 2B harmonized
    input_file: str = (
        "/home/soulios/git-soulios/DEEPFPLEARN/dfpl_DBN/data/D_fragment.csv.tsv"
    )
    # file containing the input for the harmonized molecular representations,tanimoto similarity and levenstein distance
    output_file: str = (
        "/home/soulios/git-soulios/DEEPFPLEARN/dfpl_DBN/data/D_dataset.harmonized.csv"
    )
    input_column: str = choice("smiles", "inchikey", default="inchikey")
    # column name of the column that contains the molecular representation to harmonize
    input_type: str = choice(
        "smiles", "inchikey", default="inchikey"
    )  # the type of molecular representation of the input
    output_type: str = choice(
        "smiles", "inchikey", default="smiles"
    )  # the type of molecular representation of the output


def createCommandlineParser():
    """
    Build the cmd_parser for arguments
    Parse the input arguments.
    :return: Tuple of commandlineparser, arguments
    """
    cmd_parser = ArgumentParser(prog="harmonize_mol_representations")
    cmd_parser.add_arguments(ProgramConfig, dest="config")
    cmd_args, unknown = cmd_parser.parse_known_args()
    return cmd_parser, cmd_args


def getLevensteinDistance(a, b):
    """
    This function will calculate the levenshtein distance between two input
    strings a and b

    params:
        a (String) : The first string you want to compare
        b (String) : The second string you want to compare

    returns:
        This function will return the distnace between string a and b.

    example:
        a = 'stamp'
        b = 'stomp'
        getLevensteinDistance(a,b)
        >> 1.0
    """

    if a is None and b is not None:
        return len(b)
    if a is not None and b is None:
        return len(a)

    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):
        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),  # insert character
            min_dist(s1 + 1, s2),  # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)


def getTanimotoSimilarityFromTwoSMILES(
    smile1: str = "CC(C)C=CCCCCC(=O)NCc1ccc(c(c1)OC)O",
    smile2: str = "COC1=C(C=CC(=C1)C=O)O",
) -> float:
    if smile1 is None or smile2 is None:
        return 0

    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)
    if mol1 is None or mol2 is None:
        return 0

    fp1 = Chem.RDKFingerprint(mol1, fpSize=2048)
    fp2 = Chem.RDKFingerprint(mol2, fpSize=2048)
    if fp1 is None or fp2 is None:
        return 0

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def getHarmonizedSmiles(smiles: str = "CC(C)C=CCCCCC(=O)NCc1ccc(c(c1)OC)O") -> str:
    link = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(smiles)}/smiles"
    try:
        f = urllib.request.urlopen(link)
        myfile = f.read()
        sh = myfile.decode("UTF-8")
    except urllib.error.HTTPError:
        sh = None

    return sh


def getHarmonizedSmilesfrominchis(inchikey: str = "CUWVNOSSZYUJAE-UHFFFAOYSA-N") -> str:
    link = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(inchikey)}/smiles"
    try:
        f = urllib.request.urlopen(link)
        myfile = f.read()
        sh = myfile.decode("UTF-8")
    except urllib.error.HTTPError:
        sh = None

    return sh


if __name__ == "__main__":
    parser, args = createCommandlineParser()
    opts = args.config

    # read data of smiles and harmonized smiles
    df = pd.read_csv(filepath_or_buffer=opts.input_file)
    if opts.input_type == "smiles":
        df["smiles_harmonized"] = df[opts.input_column].apply(getHarmonizedSmiles)
        df["tanimoto_similiarity"] = df.apply(
            lambda x: getTanimotoSimilarityFromTwoSMILES(
                smile1=x[opts.input_column], smile2=x["smiles_harmonized"]
            ),
            axis=1,
        )
        df["levenstein_distance"] = df.apply(
            lambda x: getLevensteinDistance(
                a=x[opts.input_column], b=x["smiles_harmonized"]
            ),
            axis=1,
        )
    elif opts.input_type == "inchikey":
        df["smiles_harmonized"] = df[opts.input_column].apply(
            getHarmonizedSmilesfrominchis
        )

    else:
        logging.error(f"Your selected input is not supported: {opts.input_file}.")
        sys.exit("Unsupported input type. Must be either smiles or inchikey")

    df.to_csv(path_or_buf=opts.output_file)
