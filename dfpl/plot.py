import array

import matplotlib.pyplot as plt
import pandas as pd
# for NN model functions
from tensorflow.keras.callbacks import History
from matplotlib.axes import Axes


# for testing in Weights & Biases


def get_max_validation_accuracy(history: History) -> str:
    validation = smooth_curve(history.history['val_accuracy'])
    y_max = max(validation)
    return 'Max validation accuracy ≈ ' + str(round(y_max, 3) * 100) + '%'


def get_max_training_accuracy(history: History) -> str:
    training = smooth_curve(history.history['accuracy'])
    y_max = max(training)
    return 'Max training accuracy ≈ ' + str(round(y_max, 3) * 100) + '%'


def smooth_curve(points: array, factor: float = 0.75) -> array:
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def set_plot_history_data(ax: Axes, history: History, which_graph: str) -> None:
    (train, valid) = (None, None)

    if which_graph == 'acc':
        train = smooth_curve(history.history['accuracy'])
        valid = smooth_curve(history.history['val_accuracy'])

    if which_graph == 'loss':
        train = smooth_curve(history.history['loss'])
        valid = smooth_curve(history.history['val_loss'])

    # plt.xkcd() # make plots look like xkcd

    epochs = range(1, len(train) + 1)

    trim = 0  # remove first 5 epochs
    # when graphing loss the first few epochs may skew the (loss) graph

    ax.plot(epochs[trim:], train[trim:], 'dodgerblue', linewidth=15, alpha=0.1)
    ax.plot(epochs[trim:], train[trim:], 'dodgerblue', label='Training')

    ax.plot(epochs[trim:], valid[trim:], 'g', linewidth=15, alpha=0.1)
    ax.plot(epochs[trim:], valid[trim:], 'g', label='Validation')


def plot_loss(hist: History, file: str) -> None:
    # fig, ax = plt.plot()
    plt.plot(hist.epoch, smooth_curve(hist.history['loss']), 'dodgerblue', linewidth=15, alpha=0.1)
    plt.plot(hist.epoch, smooth_curve(hist.history['loss']), 'dodgerblue', label='Training')
    plt.plot(hist.epoch, smooth_curve(hist.history['val_loss']), 'g', linewidth=15, alpha=0.1)
    plt.plot(hist.epoch, smooth_curve(hist.history['val_loss']), 'g', label='Validation')
    # plt.set_ylabel('Loss')
    plt.legend(loc="upper right")
    # plt.set_xlabel('Epochs')
    plt.tight_layout()
    plt.savefig(fname=file, format='jpg')
    plt.close()


def plot_history(history: History, file: str) -> None:
    fig, (ax1, ax2) = plt.subplots(nrows=2,
                                   ncols=1,
                                   figsize=(10, 6),
                                   sharex='all',
                                   gridspec_kw={'height_ratios': [5, 2]})

    set_plot_history_data(ax1, history, 'acc')

    set_plot_history_data(ax2, history, 'loss')

    # Accuracy graph
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(bottom=0.5, top=1)
    ax1.legend(loc="lower right")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    ax1.spines['bottom'].set_visible(False)

    # max accuracy text
    plt.text(0.5,
             0.6,
             get_max_validation_accuracy(history),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax1.transAxes,
             fontsize=12)
    plt.text(0.5,
             0.8,
             get_max_training_accuracy(history),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax1.transAxes,
             fontsize=12)

    # Loss graph
    ax2.set_ylabel('Loss')
    ax2.set_yticks([])
    ax2.plot(legend=False)
    ax2.set_xlabel('Epochs')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(fname=file, format='svg')
    plt.close()


def plotTrainHistory(hist, target, file_accuracy, file_loss):
    """
    Plot the training performance in terms of accuracy and loss values for each epoch.
    :param hist: The history returned by model.fit function
    :param target: The name of the target of the model
    :param file_accuracy: The filename for plotting accuracy values
    :param file_loss: The filename for plotting loss values
    :return: none
    """

    # plot accuracy
    plt.figure()
    plt.plot(hist.history['accuracy'])
    if 'val_accuracy' in hist.history.keys():
        plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy - ' + target)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if 'val_accuracy' in hist.history.keys():
        plt.legend(['Train', 'Test'], loc='upper left')
    else:
        plt.legend(['Train'], loc='upper_left')
    plt.savefig(fname=file_accuracy, format='svg')

    # Plot training & validation loss values
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss - ' + target)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    #        plt.show()
    plt.savefig(fname=file_loss, format='svg')
    plt.close()


def plot_history_vis(hist: History, model_hist_plot_path: str, model_hist_csv_path: str,
                     model_hist_plot_path_a: str, model_hist_plot_path_l: str, target: str) -> None:
    plot_history(history=hist, file=model_hist_plot_path)
    histDF = pd.DataFrame(hist.history)
    histDF.to_csv(model_hist_csv_path)

    # plot accuracy and loss for the training and validation during training
    plotTrainHistory(hist=hist, target=target,
                     file_accuracy=model_hist_plot_path_a,
                     file_loss=model_hist_plot_path_l)


def plot_auc(fpr: array, tpr: array, auc_value: float, target: str, filename: str) -> None:
    """
    Plot the area under the curve to the provided file

    :param fpr: An array containing the false positives
    :param tpr: An array containing the true positives
    :param auc_value: The value of the area under the curve
    :param target: The name of the training target
    :param filename: The filename to which the plot should be stored
    :rtype: None
    """
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_value))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve ' + target)
    plt.legend(loc='best')
    plt.savefig(fname=filename, format='svg')
    plt.close()
