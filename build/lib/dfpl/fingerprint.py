# -*- coding: utf-8 -*-
"""Calculate fingerprints"""
import logging
import multiprocessing
import os
from functools import partial
from os.path import isfile, join
from typing import Any, Callable, List

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

# import settings
from dfpl import settings

default_fp_size = 2048


def addFPColumn(data_frame: pd.DataFrame, fp_size: int) -> pd.DataFrame:
    """
    Adds a fingerprint to each row in the dataframe. This function works on
    parallel chunks of the original dataframe.
    :param data_frame: Input dataframe that needs to have a "smiles" or an "inchi" column
    :param fp_size: Number of bits in the fingerprint
    :return: The dataframe with an additional "fp" column
    """

    def smile2fp(smile: str) -> Any:
        """
        Calculates one fingerprint from a SMILE
        :param smile: Input SMILE
        :return: List of bits if conversion is successfull,
        None otherwise
        """

        npa = np.zeros((0,), dtype=np.bool)
        try:
            DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(smile), 2, nBits=fp_size
                ),
                npa,
            )
            return npa
        except:
            return None

    def inchi2fp(inchi: str) -> Any:
        """
        Calculates one fingerprint from InChI
        :param inchi: Input InChI
        :return: List of bits if conversion is successfull,
        None otherwise
        """
        try:
            return np.array(
                Chem.RDKFingerprint(Chem.MolFromInchi(inchi), fpSize=fp_size),
                dtype=settings.df_fp_numpy_type,
                copy=settings.numpy_copy_values,
            )
        except:
            # Note: We don't need to log here since rdkit already logs
            return None

    if "smiles" in data_frame:
        func = smile2fp
        accessor = "smiles"
    else:
        if "inchi" in data_frame:
            func = inchi2fp
            accessor = "inchi"
        else:
            raise ValueError("Neither smiles nor inchi column in data-frame")

    data_frame["fp"] = data_frame[accessor].apply(func)
    return data_frame


def importDataFile(
    file_name: str,
    import_function: Callable[[str], pd.DataFrame] = pd.read_csv,
    fp_size: int = default_fp_size,
) -> pd.DataFrame:
    """
    Reads data as CSV or TSV and calculates fingerprints from the SMILES in the data.
    :param import_function:
    :param file_name: Filename of CSV files containing the training data. The
        SMILES/Fingerprints are stored 1st column
    :param fp_size: Number of bits in the fingerprint
    :return: Two pandas dataframe containing the X and Y matrix for training and/or prediction. If
        no outcome data is provided, the Y matrix is a None object.
    """
    # Read the data as Pandas pickle which already contains the calculated fingerprints
    name, ext = os.path.splitext(file_name)
    if ext == ".pkl":
        return pd.read_pickle(file_name)

    df = import_function(file_name)

    # disable the rdkit logger. We know that some inchis will fail and we took care of it. No use to spam the console
    RDLogger.DisableLog("rdApp.*")
    n_cores = multiprocessing.cpu_count()
    df_split = np.array_split(df, n_cores)
    with multiprocessing.Pool(n_cores) as pool:
        df = pd.concat(pool.map(partial(addFPColumn, fp_size=fp_size), df_split))
        pool.close()
        pool.join()
    return df


def importSmilesCSV(csvfilename: str) -> pd.DataFrame:
    return pd.read_csv(csvfilename)


def importDstoxTSV(tsvfilename: str) -> pd.DataFrame:
    return pd.read_table(tsvfilename, names=["toxid", "inchi", "key"])


conversion_rules = {
    # "S_dataset.csv": importSmilesCSV,
    # "S_dataset_extended.csv": importSmilesCSV,
    # "D_dataset.tsv": importDstoxTSV,
    # "train_data.csv": importSmilesCSV,
    # "predict_data.csv": importDstoxTSV,
    "B_data_ER.csv": importDstoxTSV
}


def convert_all(directory: str) -> List[str]:
    files = [
        f
        for f in os.listdir(directory)
        for key, value in conversion_rules.items()
        if isfile(join(directory, f)) and f in key
    ]
    logging.info(f"Found {len(files)} files to convert")
    for f in files:
        path = join(directory, f)
        logging.info(f"Importing file {f}")
        df = importDataFile(path, import_function=conversion_rules[f])
        name, ext = os.path.splitext(f)
        output_file = join(directory, name + ".pkl")
        logging.info(f"Saving pickle of file {f}")
        df.to_pickle(output_file)
    return files
