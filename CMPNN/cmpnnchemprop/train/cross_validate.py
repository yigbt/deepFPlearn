from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import csv
from typing import Callable, Dict, List, Tuple
from .run_training import run_training
from cmpnnchemprop.data.utils import get_task_names
from cmpnnchemprop.utils import makedirs


def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores, scores_and_metrics = run_training(args, logger)
        all_scores.append(model_scores)
        with open(os.path.join(args.save_dir, 'scores.csv'), 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["set", "metric", "score", "epoch"])
            csv_writer.writerows(scores_and_metrics)
        scoresdf = pd.read_csv(f"{args.save_dir}/scores.csv")
        grouped = scoresdf.groupby(["metric", "set"])
        plt.clf()
            # Iterate through the groups and create a line plot for each
        for (metric, set), group in grouped:
            plt.plot(group["epoch"], group["score"], label=f"{set} {metric}")
            plt.xlabel("epoch")
            plt.ylabel("score")
            plt.legend()
            plt.savefig(f"{args.save_dir}/scores_and_metrics.png") 
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score
