import csv
import logging
import math
import os.path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import tensorflow.keras.metrics as metrics
import wandb
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.python.framework.ops import disable_eager_execution

from dfpl import callbacks
from dfpl import history as ht
from dfpl import options, settings
from dfpl.autoencoder import create_dense_layer, setup_train_test_split
from dfpl.utils import ae_scaffold_split, weight_split

disable_eager_execution()


def define_vae_model(opts: options.Options) -> Tuple[Model, Model]:
    """
    Define and compile a Variational Autoencoder (VAE) model based on the given options.

    :param opts: Training options with model parameters.
    :param output_bias: Bias for the output layer, used for initializing the last layer.
    :return: Tuple containing the VAE and encoder models.
    """
    input_size = opts.fpSize
    encoding_dim = opts.encFPSize

    lr_schedule = optimizers.schedules.ExponentialDecay(
        opts.aeLearningRate,
        decay_steps=1000,
        decay_rate=opts.aeLearningRateDecay,
        staircase=True,
    )
    vae_optimizer = optimizers.legacy.Adam(learning_rate=lr_schedule)
    input_vec = Input(shape=(input_size,))
    initial_layer_size = int(input_size / 2)
    encoded = create_dense_layer(
        input_vec, initial_layer_size, opts.aeActivationFunction
    )

    # Start `layer_sizes` with the initial layer size (1024)
    layer_sizes = [initial_layer_size]

    # Building encoding layers and storing their sizes
    hidden_layer_count = round(math.log2(input_size / encoding_dim))
    for i in range(1, hidden_layer_count - 1):
        layer_size = int(input_size / (2 ** (i + 1)))
        layer_sizes.append(layer_size)
        encoded = create_dense_layer(encoded, layer_size, opts.aeActivationFunction)

    # Latent space layers
    z_mean = create_dense_layer(encoded, encoding_dim, opts.aeActivationFunction)
    z_log_var = create_dense_layer(encoded, encoding_dim, opts.aeActivationFunction)
    # Sampling layer

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(opts.vaeBeta * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(encoding_dim,))([z_mean, z_log_var])

    # Build decoder layers starting directly from `z`
    decoded = z
    for layer_size in reversed(layer_sizes):
        decoded = create_dense_layer(decoded, layer_size, opts.aeActivationFunction)

    # Final output layer to reconstruct input
    decoded = Dense(units=input_size, activation="sigmoid")(decoded)

    # Define VAE and encoder models
    vae = Model(input_vec, decoded)
    encoder = Model(input_vec, z_mean)

    # Define custom loss
    def kl_loss(z_mean, z_log_var):
        return -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
        )

    def bce_loss(y_true, y_pred):
        return metrics.binary_crossentropy(y_true, y_pred)

    def vae_loss(y_true, y_pred):
        return bce_loss(y_true, y_pred) + 0.5 * kl_loss(z_mean, z_log_var)

    vae.compile(optimizer=vae_optimizer, loss=vae_loss, metrics=[kl_loss, bce_loss])
    vae.summary(print_fn=logging.info)

    return vae, encoder


def train_full_vae(
    df: pd.DataFrame, opts: options.Options
) -> Tuple[Model, np.ndarray, np.ndarray]:
    """
    Trains a VAE on the provided data and returns the trained encoder and split indices.

    :param df: DataFrame containing SMILES/InChI data for training.
    :param opts: Training options.
    :return: The encoder model of the trained VAE and split indices.
    """
    # Initialize wandb tracking if needed
    if opts.aeWabTracking:
        wandb.init(project=f"VAE_{opts.aeSplitType}")

    # Define paths for saving weights
    save_path = os.path.join(opts.ecModelDir, f"vae_weights.h5")

    x_train, x_test, train_indices, test_indices = setup_train_test_split(df, opts)

    # Define VAE and encoder models
    vae, encoder = define_vae_model(opts)

    # Set up callbacks and train the VAE model
    callback_list = callbacks.autoencoder_callback(checkpoint_path=save_path, opts=opts)
    vae_hist = vae.fit(
        x_train,
        x_train,
        epochs=opts.aeEpochs,
        batch_size=opts.aeBatchSize,
        verbose=opts.verbose,
        callbacks=[callback_list],
        validation_data=(x_test, x_test) if x_test is not None else None,
    )

    # Store training history
    ht.store_and_plot_history(base_file_name=save_path, hist=vae_hist)

    # load the whole vae from the checkpoint
    vae.load_weights(save_path)
    encoder.save_weights(os.path.join(opts.ecModelDir, "encoder_weights.h5"))

    return encoder, train_indices, test_indices
