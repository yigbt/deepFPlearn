# for NN model functions
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# for testing in Weights & Biases
from wandb.keras import WandbCallback

from dfpl import options, settings


def autoencoder_callback(checkpoint_path: str, opts: options.Options) -> list:
    """
    Callbacks for fitting the autoencoder

    :param checkpoint_path: The output directory to store the checkpoint weight files
    :param opts: Training options provided to the run
    :return: List of ModelCheckpoint and EarlyStopping class.
    """
    callbacks = []

    if opts.testSize > 0.0:
        target = "val_loss"
    else:
        target = "loss"
        # enable this checkpoint to restore the weights of the best performing model
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor=target,
        mode="min",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        period=settings.ac_train_check_period,
    )
    callbacks.append(checkpoint)

    # enable early stopping if val_loss is not improving anymore
    early_stop = EarlyStopping(
        monitor=target,
        mode="min",
        patience=settings.ac_train_patience,
        min_delta=settings.ac_train_min_delta,
        verbose=1,
        restore_best_weights=True,
    )
    callbacks.append(early_stop)
    if opts.aeWabTracking:
        callbacks.append(WandbCallback(save_model=False))
    return callbacks


def nn_callback(checkpoint_path: str, opts: options.Options) -> list:
    """
    Callbacks for fitting the feed forward network (FNN)

    :param checkpoint_path: The output directory to store the checkpoint weight files
    :param opts: Training options provided to the run
    :return: List of ModelCheckpoint and EarlyStopping class.
    """

    callbacks = []

    if opts.testSize > 0.0:
        # enable this checkpoint to restore the weights of the best performing model
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            verbose=1,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
            period=settings.nn_train_check_period,
        )
        callbacks.append(checkpoint)

        # enable early stopping if val_loss is not improving anymore
        early_stop = EarlyStopping(
            patience=settings.nn_train_patience,
            monitor="val_loss",
            mode="min",
            min_delta=settings.nn_train_min_delta,
            verbose=1,
            restore_best_weights=True,
        )
        callbacks.append(early_stop)
    if opts.wabTracking:
        callbacks.append(WandbCallback(save_model=False))
    return callbacks
