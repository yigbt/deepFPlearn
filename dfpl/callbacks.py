# for NN model functions
from keras.callbacks import ModelCheckpoint, EarlyStopping
# for testing in Weights & Biases
from wandb.keras import WandbCallback

from dfpl import options
from dfpl import settings


def nn_callback(checkpoint_path: str, opts: options.TrainOptions) -> list:
    """
    Callbacks for fitting the autoencoder

    :param checkpoint_path: The output directory to store the checkpoint weight files
    :return: List of ModelCheckpoint and EarlyStopping class.
    """

    # enable this checkpoint to restore the weights of the best performing model
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 verbose=1,
                                 period=settings.nn_train_check_period,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)

    # enable early stopping if val_loss is not improving anymore
    early_stop = EarlyStopping(patience=settings.nn_train_patience,
                               monitor="loss",
                               min_delta=settings.nn_train_min_delta,
                               verbose=1,
                               restore_best_weights=True)

    if opts.wabTracking:
        trackWandB_callback = WandbCallback(save_model=False)
        return [checkpoint, early_stop, trackWandB_callback]
    else:
        return [checkpoint, early_stop]
