# -*- coding: utf-8 -*-
"""Store and visualise training histories"""

import pandas as pd
import logging
from tensorflow.keras.callbacks import History
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# def store_and_plot_history(base_file_name: str, hist: History) -> None:
#     """

#     :param base_file_name:
#     :param hist:
#     :return:
#     """

#     (ac_history_csv, ac_history_svg) = (base_file_name + ".history.csv",
#                                         base_file_name + ".history.svg")

#     # store history
#     pd.DataFrame(hist.history).to_csv(ac_history_csv)
#     logging.info(f"Neural network training history saved in file: {ac_history_csv}")

#     # plot history
#     ac_epochs = hist.epoch

#     # generate a figure of the losses for this fold
#     plt.figure()
#     for k in hist.history.keys():
#         plt.plot(ac_epochs, hist.history[k], label=k)
#     plt.title('Training and validation metrics of neural network')
#     plt.legend()
#     plt.savefig(fname=ac_history_svg,
#                 format='svg')
#     plt.close()
#     logging.info(f"Neural network training history plotted in file: {ac_history_svg}")
def store_and_plot_history(base_file_name: str, hist: History) -> None:
    """
    Store training history to file and plot history of loss and AUC.

    :param base_file_name: Base name of the file to store the history and plot to.
    :param hist: Keras history object.
    """
    # Define file names for CSV and SVG files
    csv_filename = base_file_name + ".history.csv"
    svg_filename = base_file_name + ".history.svg"

    # Store history as CSV file
    pd.DataFrame(hist.history).to_csv(csv_filename)
    logging.info(f"Neural network training history saved in file: {csv_filename}")

    # Plot history of loss and AUC
    epochs = range(1, len(hist.history['loss']) + 1)
    fig, (ax2, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    plt.suptitle('Training and validation metrics of neural network')
    
    # Plot loss and validation loss
    ax1.plot(epochs, hist.history['loss'], linestyle="--", label='Training loss',color = "blue")
    ax1.plot(epochs, hist.history['val_loss'], label='Validation loss',color = "blue")
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot AUC and validation AUC
    ax2.plot(epochs, hist.history['auc'],linestyle ="--" , label='Training AUC',color = "blue")
    ax2.plot(epochs, hist.history['val_auc'], label='Validation AUC',color = "blue")
    ax2.set_ylabel('AUC')
    ax2.set_xlabel('Epoch')
    ax2.legend()


    # Save and close figure
    plt.savefig(fname=svg_filename, format='svg')
    plt.close()
    logging.info(f"Neural network training history plotted in file: {svg_filename}")


