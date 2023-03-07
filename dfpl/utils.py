import pathlib
import os
from matplotlib import pyplot as plt
import json
import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict
import logging
from random import Random
from typing import Dict, List, Set, Tuple, Union
import warnings
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np
import random
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def makePathAbsolute(p: str) -> str:
    path = pathlib.Path(p)
    if path.is_absolute():
        return p
    else:
        return str(path.absolute())


def createDirectory(directory: str):
    path = makePathAbsolute(directory)
    if not os.path.exists(path):
        os.makedirs(path)


# def makePlots(save_path: str, training_auc: list, training_loss: list, validation_auc: list, validation_loss: list):
#
#     all_data = [training_auc, training_loss, validation_auc, validation_loss]
#     zipped = list(zip(training_loss, validation_loss,
#                   training_auc, validation_auc))
#
#     metricsdf = pd.DataFrame(
#         zipped, columns=['LOSS', 'VAL_LOSS', 'AUC', 'VAL_AUC'])
#     metricsdf.to_csv(f"{save_path}/metrics.csv")
#
#     metricsdf.plot(title='Model performance')
#     plt.savefig(f"{save_path}/plot.png", format='png')
#
#     lossesdf = pd.DataFrame(
#         list(zip(training_loss, validation_loss)), columns=['LOSS', 'VAL_LOSS'])
#     lossesdf.plot(title='Loss')
#     plt.savefig(f"{save_path}/loss.png", format='png')
#
#     aucdf = pd.DataFrame(
#         list(zip(training_auc, validation_auc)), columns=['AUC', 'VAL_AUC'])
#     aucdf.plot(title='AUC')
#     plt.savefig(f"{save_path}/auc.png", format='png')


def createArgsFromJson(in_json: str, ignore_elements: list, return_json_object: bool):
    arguments = []
    with open(in_json, 'r') as f:
        data = json.load(f)
    for i, j in data.items():
        if str(i) not in ignore_elements:
            i = "--" + str(i)
            j = str(j)
            arguments.append(i)
            arguments.append(j)
    if return_json_object:
        return arguments, data
    return arguments


def make_mol(s: str, keep_h: bool, add_h: bool, keep_atom_map: bool):
    """
    Builds an RDKit molecule from a SMILES string.

    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h if not keep_atom_map else False
    mol = Chem.MolFromSmiles(s, params)

    if add_h:
        mol = Chem.AddHs(mol)

    if keep_atom_map:
        atom_map_numbers = tuple(atom.GetAtomMapNum()
                                 for atom in mol.GetAtoms())
        for idx, map_num in enumerate(atom_map_numbers):
            if idx + 1 != map_num:
                new_order = np.argsort(atom_map_numbers).tolist()
                return Chem.rdmolops.RenumberAtoms(mol, new_order)

    return mol


def generate_scaffold(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    if isinstance(mol, str):
        mol = make_mol(mol, keep_h=False, add_h=False, keep_atom_map=False)
    if isinstance(mol, tuple):
        mol = mol[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_to_smiles(mols: List[str], use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param mols: A list of SMILES.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, smiles in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(smiles)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(smiles)

    return scaffolds


def scaffold_split(data: pd.DataFrame,
                   sizes: Tuple[float, float, float] = (0.8, 0, 0.2),
                   balanced: bool = False,
                   key_molecule_index: int = 0,
                   seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a pandas DataFrame by scaffold so that no molecules sharing a scaffold are in different splits.
    :param data: A pandas DataFrame containing SMILES strings and molecule properties.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param balanced: Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    :param key_molecule_index: For data with multiple molecules, this sets which molecule will be considered during splitting.
    :param seed: Random seed for shuffling when doing balanced splitting.
    :return: A tuple of pandas DataFrames containing the train, validation, and test splits of the data.
    """
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Invalid train/val/test splits! got: {sizes}")

    # Split
    train_size, val_size, test_size = sizes[0] * \
        len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    key_mols = data.iloc[:, key_molecule_index]
    scaffold_to_indices = scaffold_to_smiles(
        key_mols.tolist(), use_indices=True)
    # Seed randomness
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)
    for index_set in index_sets:
        if len(test) + len(index_set) <= test_size:
            test += index_set
            test_scaffold_count += 1
        elif len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        else:
            val += index_set
            val_scaffold_count += 1
    logging.info(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                 f'train scaffolds = {train_scaffold_count:,} | '
                 f'val scaffolds = {val_scaffold_count:,} | '
                 f'test scaffolds = {test_scaffold_count:,}')

    log_scaffold_stats(data, index_sets)
    # Map from indices to data

    train_df = data.iloc[train]
    val_df = data.iloc[val]
    test_df = data.iloc[test]
    return train_df, val_df, test_df



def log_scaffold_stats(data: pd.DataFrame,
                       index_sets: List[Set[int]],
                       num_scaffolds: int = 10,
                       num_labels: int = 20) -> List[Tuple[List[float], List[int]]]:
    """
    Logs and returns statistics about counts and average target values in molecular scaffolds.
    :param data: A pandas DataFrame containing SMILES strings and molecule properties.
    :param index_sets: A list of sets of indices representing splits of the data.
    :param num_scaffolds: The number of scaffolds about which to display statistics.
    :param num_labels: The number of labels about which to display statistics.
    :return: A list of tuples where each tuple contains a list of average target values
    across the first :code:num_labels labels and a list of the number of non-zero values for
    the first :code:num_scaffolds scaffolds, sorted in decreasing order of scaffold frequency.
    """
    logging.info('Label averages per scaffold, in decreasing order of scaffold frequency, '
                     f'capped at {num_scaffolds} scaffolds and {num_labels} labels:')

    stats = []
    index_sets = sorted(
        index_sets, key=lambda idx_set: len(idx_set), reverse=True)
    for scaffold_num, index_set in enumerate(index_sets[:num_scaffolds]):
        data_set = data.iloc[list(index_set)]
        targets = [ c for c in data.columns if c in ['AR', 'ER', 'ED', 'TR', 'GR','PPARg','Aromatase']]
        # targets = data_set.iloc[:, 2:].values
        targets = data_set.loc[:, targets].values

        with warnings.catch_warnings():  # Likely warning of empty slice of target has no values besides NaN
            warnings.simplefilter('ignore', category=RuntimeWarning)
            target_avgs = np.nanmean(targets, axis=0)[:num_labels]

        counts = np.count_nonzero(~np.isnan(targets), axis=0)[:num_labels]
        stats.append((target_avgs, counts))

        logging.info(f'Scaffold {scaffold_num}')
        for task_num, (target_avg, count) in enumerate(zip(target_avgs, counts)):
            logging.info(
                f'Task {task_num}: count = {count:,} | target average = {target_avg:.6f}')
        logging.info('\n')
    return stats
