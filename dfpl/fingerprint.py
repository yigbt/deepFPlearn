# -*- coding: utf-8 -*-
"""Calculates fingerprints"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from typing import Any
import multiprocessing
from functools import partial
from typing import Callable

default_fp_size = 2048


def addFPColumn(data_frame: pd.DataFrame, fp_size: int) -> pd.DataFrame:
    """
    Adds a fingerprint to each row in the dataframe. This function is
    intended to be called in parallel chunks of the original dataframe.

    :param data_frame: Input dataframe that needs to have a "smiles"
    or a "inchi" column
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
        try:
            return np.array(
                Chem.RDKFingerprint(Chem.MolFromSmiles(smile), fpSize=fp_size))
        except:
            # Note: We don't need to log here since rdkit already logs
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
                Chem.RDKFingerprint(Chem.MolFromInchi(inchi),
                                    fpSize=fp_size))
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


def processInParallel(
        csvfilename: str,
        import_function: Callable[[str], pd.DataFrame] = pd.read_csv,
        fp_size: int = default_fp_size) -> pd.DataFrame:
    """
    Return the matrix of features for training and testing NN models (X) as numpy array.
    Provided SMILES are transformed to fingerprints, fingerprint strings are then split
    into vectors and added as row to the array which is returned.
    :param import_function:
    :param csvfilename: Filename of CSV files containing the training data. The
        SMILES/Fingerprints are stored 1st column
    :param fp_size: Number of bits in the fingerprint
    :return: Two pandas dataframe containing the X and Y matrix for training and/or prediction. If
        no outcome data is provided, the Y matrix is a None object.
    """
    df = import_function(csvfilename)

    RDLogger.DisableLog("rdApp.*")
    n_cores = multiprocessing.cpu_count()
    df_split = np.array_split(df, n_cores)
    with multiprocessing.Pool(n_cores) as pool:
        df = pd.concat(
            pool.map(partial(addFPColumn, fp_size=fp_size), df_split))
        pool.close()
        pool.join()
    return df


def importSmilesCSV(csvfilename: str) -> pd.DataFrame:
    return pd.read_csv(csvfilename)


def importDstoxTSV(tsvfilename: str) -> pd.DataFrame:
    return pd.read_table(tsvfilename, names=["toxid", "inchi", "key"])
