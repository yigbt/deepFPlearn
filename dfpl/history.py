# -*- coding: utf-8 -*-
"""Store and visualise training histories"""

import pandas as pd
import logging
from tensorflow.keras.callbacks import History
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def store_and_plot_history(base_file_name: str, hist: History) -> None:
    """

    :param base_file_name:
    :param hist:
    :return:
    """

    (ac_history_csv, ac_history_svg) = (base_file_name + ".history.csv",
                                        base_file_name + ".history.svg")

    # store history
    pd.DataFrame(hist.history).to_csv(ac_history_csv)
    logging.info(f"Neural network training history saved in file: {ac_history_csv}")

    # plot history
    ac_epochs = hist.epoch

    # generate a figure of the losses for this fold
    plt.figure()
    for k in hist.history.keys():
        plt.plot(ac_epochs, hist.history[k], label=k)
    plt.title('Training and validation metrics of neural network')
    plt.legend()
    plt.savefig(fname=ac_history_svg,
                format='svg')
    plt.close()
    logging.info(f"Neural network training history plotted in file: {ac_history_svg}")
