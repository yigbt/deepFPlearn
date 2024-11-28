from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.axes import Axes

# for NN model functions
from tensorflow.python.keras.callbacks import History


def get_max_validation_accuracy(history: History) -> str:
    validation = smooth_curve(history.history["val_accuracy"])
    y_max: float = max(validation)
    return "Max validation accuracy ≈ " + str(round(y_max, 3) * 100) + "%"


def get_max_validation_balanced_accuracy(history: History) -> str:
    validation_bal_acc = smooth_curve(history.history["val_balanced_accuracy"])
    y_max: float = max(validation_bal_acc)
    return "Max validation balanced accuracy ≈ " + str(round(y_max, 3) * 100) + "%"


def get_max_training_balanced_accuracy(history: History) -> str:
    training_bal_acc = smooth_curve(history.history["balanced_accuracy"])
    y_max: float = max(training_bal_acc)
    return "Training balanced accuracy ≈ " + str(round(y_max, 3) * 100) + "%"


def get_max_training_auc(history: History) -> str:
    training_auc = smooth_curve(history.history["auc"])
    y_max: float = max(training_auc)
    return "Validation AUC ≈ " + str(round(y_max, 3) * 100) + "%"


def get_max_validation_auc(history: History) -> str:
    validation_auc = smooth_curve(history.history["val_auc"])
    y_max: float = max(validation_auc)
    return "Validation AUC ≈ " + str(round(y_max, 3) * 100) + "%"


def get_max_training_accuracy(history: History) -> str:
    training = smooth_curve(history.history["accuracy"])
    y_max: float = max(training)
    return "Max training accuracy ≈ " + str(round(y_max, 3) * 100) + "%"


def smooth_curve(points: np.ndarray, factor: float = 0.8) -> List[float]:
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# Plot the accuracy and loss data with enhanced visuals
def set_plot_history_data(ax: Axes, history: History, which_graph: str) -> None:
    if which_graph == "balanced_acc":
        # Plot balanced accuracy when "acc" is specified
        train = smooth_curve(np.array(history.history["balanced_accuracy"]))
        valid = smooth_curve(np.array(history.history["val_balanced_accuracy"]))
        label = "Balanced Accuracy"
    elif which_graph == "loss":
        train = smooth_curve(np.array(history.history["loss"]))
        valid = smooth_curve(np.array(history.history["val_loss"]))
        label = "Loss"
    else:
        return

    epochs = range(1, len(train) + 1)

    # Plot training and validation data with styles
    ax.plot(epochs, train, color="dodgerblue", linewidth=2, label=f"Training {label}")
    ax.plot(
        epochs,
        valid,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Validation {label}",
    )
    ax.set_ylabel(label)
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_history(history: History, file: str) -> None:
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex="all")

    set_plot_history_data(ax1, history, "balanced_acc")
    set_plot_history_data(ax2, history, "loss")

    # Set shared x-axis label and save the plot
    ax2.set_xlabel("Epochs")
    plt.tight_layout()
    plt.savefig(fname=file, format="svg")
    plt.close()


def plot_train_history(hist, target, file_accuracy, file_loss):
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
    plt.plot(hist.history["accuracy"])
    if "val_accuracy" in hist.history.keys():
        plt.plot(hist.history["val_accuracy"])
    plt.title("Model accuracy - " + target)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    if "val_accuracy" in hist.history.keys():
        plt.legend(["Train", "Test"], loc="upper left")
    else:
        plt.legend(["Train"], loc="upper_left")
    plt.savefig(fname=file_accuracy, format="svg")

    # Plot training & validation loss values
    plt.figure()
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Model loss - " + target)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    #        plt.show()
    plt.savefig(fname=file_loss, format="svg")
    plt.close()


def plot_history_vis(
    hist: History,
    model_hist_plot_path: str,
    model_hist_csv_path: str,
    model_hist_plot_path_a: str,
    model_hist_plot_path_l: str,
    target: str,
) -> None:
    plot_history(history=hist, file=model_hist_plot_path)
    histDF = pd.DataFrame(hist.history)
    histDF.to_csv(model_hist_csv_path)

    # plot accuracy and loss for the training and validation during training
    plot_train_history(
        hist=hist,
        target=target,
        file_accuracy=model_hist_plot_path_a,
        file_loss=model_hist_plot_path_l,
    )


# Enhanced AUC plot
def plot_auc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_value: float,
    target: str,
    filename: str,
    wandb_logging: bool = False,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"AUC = {auc_value:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {target}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(fname=filename, format="png")
    if wandb_logging:
        wandb.log({"roc_plot": plt})
    plt.close()
