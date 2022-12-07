import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os.path
from os.path import basename
import math
import numpy as np
import pandas as pd
import logging
import argparse
import os
import tensorflow.keras.metrics as metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers, losses, initializers
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
from dfpl import options
from dfpl import callbacks
from dfpl import history as ht
from dfpl import settings
# import chemprop as cp
from cmpnnchemprop.train.run_training import run_training
from cmpnnchemprop.data.utils import get_task_names
from cmpnnchemprop.utils import makedirs
# from chemprop.parsing import parse_predict_args
# from chemprop.train import make_predictions
from matplotlib import pyplot as plt

def cross_validate(opts = options.GnnOptions, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = opts.seed
    save_dir = opts.save_dir
    task_names = get_task_names(opts.data_path)
    # Run training on different random seeds for each fold
    all_scores = []
    save_path = opts.save_dir
    for fold_num in range(opts.num_folds):
        info(f'Fold {fold_num}')
        opts.seed = init_seed + fold_num
        opts.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(opts.save_dir)
        model_scores = run_training(opts, logger)
        all_scores.append(model_scores)


    all_scores = np.array(all_scores)
    # Report results
    info(f'{opts.num_folds}-fold cross validation')

    # Report scores for each fold
    all_fold_score = []
    for fold_num, scores in enumerate(all_scores):
        info(f'Fold {fold_num} ==> test {opts.metric} = {np.nanmean(scores):.6f}')
        all_fold_score.append(np.nanmean(scores))

        if opts.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {opts.metric} = {score:.6f}')

    max_index = all_fold_score.index(max(all_fold_score))
    print("All Fold List ===", all_fold_score)
    print("max index====", max_index)
    # plot_path = os.path.join(save_path, f'fold_{max_index}', 'plot.png')
    # new_name = plot_path.replace("plot", "best_plot")
    # os.rename(plot_path, new_name)

    plot_path = os.path.join(save_path, f'fold_{max_index}', 'plot.png')
    # new_name = plot_path.replace("plot", "best_plot")
    new_name = f'best_plot_fold_{max_index}'
    new_name = plot_path.replace("plot", new_name)
    os.rename(plot_path, new_name)

    cmd = f'mv {new_name} {save_path}/'
    os.system(cmd)





    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {opts.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if opts.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {opts.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score


