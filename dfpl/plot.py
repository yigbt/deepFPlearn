from typing import List

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.axes import Axes
from sklearn.metrics import auc, precision_recall_curve
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
    smoothed_points: List[float] = []
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


def plot_prc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target: str,
    filename: str,
    wandb_logging: bool = False,
) -> None:
    """
    Plot the Precision-Recall Curve (PRC) with AUC.

    :param y_true: True binary labels
    :param y_scores: Target scores, typically predicted probabilities
    :param target: The name of the model or target being evaluated
    :param filename: The filename to save the plot
    :param wandb_logging: Whether to log the plot to Weights & Biases
    :rtype: None
    """
    # Calculate precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    prc_auc_value = auc(recall, precision)

    # Plot PRC curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="purple",
        linewidth=2,
        label=f"PRC-AUC = {prc_auc_value:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {target}")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save plot
    plt.savefig(fname=filename, format="png")
    if wandb_logging:
        wandb.log({"prc_plot": plt})
    plt.close()
