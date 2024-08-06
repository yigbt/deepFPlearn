import logging
import math
import os.path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow.keras.metrics as metrics
import wandb
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers, optimizers
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.framework.ops import disable_eager_execution

from dfpl.train import TrainOptions
from dfpl import callbacks
from dfpl import history as ht
from dfpl import settings
from dfpl.utils import ae_scaffold_split, weight_split

disable_eager_execution()


def define_vae_model(opts: TrainOptions, output_bias=None) -> Tuple[Model, Model]:
    input_size = opts.fpSize
    encoding_dim = (
        opts.encFPSize
    )  # This should be the intended size of your latent space, e.g., 256

    lr_schedule = optimizers.schedules.ExponentialDecay(
        opts.aeLearningRate,
        decay_steps=1000,
        decay_rate=opts.aeLearningRateDecay,
        staircase=True,
    )
    ac_optimizer = optimizers.legacy.Adam(learning_rate=lr_schedule)

    if output_bias is not None:
        output_bias = initializers.Constant(output_bias)

    hidden_layer_count = round(math.log2(input_size / encoding_dim))

    input_vec = Input(shape=(input_size,))

    # 1st hidden layer
    if opts.aeActivationFunction != "selu":
        encoded = Dense(
            units=int(input_size / 2), activation=opts.aeActivationFunction
        )(input_vec)
    else:
        encoded = Dense(
            units=int(input_size / 2),
            activation=opts.aeActivationFunction,
            kernel_initializer="lecun_normal",
        )(input_vec)

    # encoding layers
    for i in range(
        1, hidden_layer_count - 1
    ):  # Adjust the range to stop before the latent space layers
        factor_units = 2 ** (i + 1)
        if opts.aeActivationFunction != "selu":
            encoded = Dense(
                units=int(input_size / factor_units),
                activation=opts.aeActivationFunction,
            )(encoded)
        else:
            encoded = Dense(
                units=int(input_size / factor_units),
                activation=opts.aeActivationFunction,
                kernel_initializer="lecun_normal",
            )(encoded)

    # latent space layers
    if opts.aeActivationFunction != "selu":
        z_mean = Dense(units=encoding_dim, activation=opts.aeActivationFunction)(
            encoded
        )  # Adjusted size to encoding_dim
        z_log_var = Dense(units=encoding_dim, activation=opts.aeActivationFunction)(
            encoded
        )  # Adjusted size to encoding_dim
    else:
        z_mean = Dense(
            units=encoding_dim,
            activation=opts.aeActivationFunction,
            kernel_initializer="lecun_normal",
        )(
            encoded
        )  # Adjusted size to encoding_dim
        z_log_var = Dense(
            units=encoding_dim,
            activation=opts.aeActivationFunction,
            kernel_initializer="lecun_normal",
        )(
            encoded
        )  # Adjusted size to encoding_dim

    # sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(encoding_dim,))([z_mean, z_log_var])
    decoded = z

    # decoding layers
    for i in range(hidden_layer_count - 2, 0, -1):
        factor_units = 2**i
        if opts.aeActivationFunction != "selu":
            decoded = Dense(
                units=int(input_size / factor_units),
                activation=opts.aeActivationFunction,
            )(decoded)
        else:
            decoded = Dense(
                units=int(input_size / factor_units),
                activation=opts.aeActivationFunction,
                kernel_initializer="lecun_normal",
            )(decoded)

    # output layer
    decoded = Dense(
        units=input_size, activation="sigmoid", bias_initializer=output_bias
    )(decoded)

    autoencoder = Model(input_vec, decoded)
    encoder = Model(input_vec, z)
    autoencoder.summary(print_fn=logging.info)

    # KL divergence loss
    def kl_loss(z_mean, z_log_var):
        return -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
        )

    # binary cross-entropy loss
    def bce_loss(y_true, y_pred):
        return metrics.binary_crossentropy(y_true, y_pred)

    # combined loss
    def vae_loss(y_true, y_pred):
        bce = bce_loss(y_true, y_pred)
        kl = kl_loss(z_mean, z_log_var)
        return bce + 0.5 * kl

    autoencoder.compile(
        optimizer=ac_optimizer, loss=vae_loss, metrics=[bce_loss, kl_loss]
    )

    return autoencoder, encoder


def train_full_vae(df: pd.DataFrame, opts: TrainOptions) -> Model:
    """
    Trains an autoencoder on the given feature matrix X. The response matrix is only used to
    split the data into meaningful test and train sets.

    :param opts: Command line arguments
    :param df: Pandas dataframe that contains the SMILES/InChI data for training the autoencoder
    :return: The encoder model of the trained autoencoder
    """

    # If wandb tracking is enabled for VAE weights but not for the main program, initialize a new wandb run
    if opts.aeWabTracking and not opts.wabTracking:
        wandb.init(project=f"VAE_{opts.aeSplitType}")

    save_path = os.path.join(opts.ecModelDir, f"{opts.aeSplitType}_split_autoencoder")
    # Collect the callbacks for training

    # Select all fingerprints that are valid and turn them into a numpy array
    fp_matrix = np.array(
        df[df["fp"].notnull()]["fp"].to_list(),
        dtype=settings.ac_fp_numpy_type,
        copy=settings.numpy_copy_values,
    )
    logging.info(
        f"Training VAE on a matrix of shape {fp_matrix.shape} with type {fp_matrix.dtype}"
    )
    assert 0.0 <= opts.testSize <= 0.5
    if opts.aeSplitType == "random":
        logging.info("Training autoencoder using random split")
        initial_indices = np.arange(fp_matrix.shape[0])
        if opts.testSize > 0.0:
            # Split data into test and training data
            if opts.aeWabTracking:
                x_train, x_test, train_indices, test_indices = train_test_split(
                    fp_matrix, initial_indices, test_size=opts.testSize, random_state=42
                )
            else:
                x_train, x_test, train_indices, test_indices = train_test_split(
                    fp_matrix, initial_indices, test_size=opts.testSize, random_state=42
                )
        else:
            x_train = fp_matrix
            x_test = None
    elif opts.aeSplitType == "scaffold_balanced":
        logging.info("Training autoencoder using scaffold split")
        train_indices = np.arange(fp_matrix.shape[0])
        if opts.testSize > 0.0:
            # if opts.aeWabTracking:
            train_data, val_data, test_data = ae_scaffold_split(
                df,
                sizes=(1 - opts.testSize, 0.0, opts.testSize),
                balanced=True,
                seed=42,
            )
            x_train = np.array(
                train_data[train_data["fp"].notnull()]["fp"].to_list(),
                dtype=settings.ac_fp_numpy_type,
                copy=settings.numpy_copy_values,
            )
            x_test = np.array(
                test_data[test_data["fp"].notnull()]["fp"].to_list(),
                dtype=settings.ac_fp_numpy_type,
                copy=settings.numpy_copy_values,
            )
            train_indices = df[
                df.index.isin(train_data[train_data["fp"].notnull()].index)
            ].index.to_numpy()
            test_indices = df[
                df.index.isin(test_data[test_data["fp"].notnull()].index)
            ].index.to_numpy()
        else:
            x_train = fp_matrix
            x_test = None
    elif opts.aeSplitType == "molecular_weight":
        logging.info("Training autoencoder using molecular weight split")
        train_indices = np.arange(fp_matrix.shape[0])
        if opts.testSize > 0.0:
            train_data, val_data, test_data = weight_split(
                df, sizes=(1 - opts.testSize, 0.0, opts.testSize), bias="small"
            )
            x_train = np.array(
                train_data[train_data["fp"].notnull()]["fp"].to_list(),
                dtype=settings.ac_fp_numpy_type,
                copy=settings.numpy_copy_values,
            )
            x_test = np.array(
                test_data[test_data["fp"].notnull()]["fp"].to_list(),
                dtype=settings.ac_fp_numpy_type,
                copy=settings.numpy_copy_values,
            )
            df_sorted = df.sort_values(by="mol_weight", ascending=True)
            # Get the sorted indices from the sorted DataFrame
            sorted_indices = df_sorted.index.to_numpy()

            # Find the corresponding indices for train_data, val_data, and test_data in the sorted DataFrame
            train_indices = sorted_indices[df.index.isin(train_data.index)]
            # val_indices = sorted_indices[df.index.isin(val_data.index)]
            test_indices = sorted_indices[df.index.isin(test_data.index)]
        else:
            x_train = fp_matrix
            x_test = None
    else:
        raise ValueError(f"Invalid split type: {opts.split_type}")

    # Calculate the initial bias aka the log ratio between 1's and 0'1 in all fingerprints
    ids, counts = np.unique(x_train.flatten(), return_counts=True)
    count_dict = dict(zip(ids, counts))
    if count_dict[0] == 0:
        initial_bias = None
        logging.info("No zeroes in training labels. Setting initial_bias to None.")
    else:
        initial_bias = np.log([count_dict[1] / count_dict[0]])
        logging.info(f"Initial bias for last sigmoid layer: {initial_bias[0]}")
    if opts.testSize > 0.0:
        logging.info(f"VAE training/testing mode with train- and test-samples")
        logging.info(f"VAE train data shape {x_train.shape} with type {x_train.dtype}")
        logging.info(f"VAE test data shape {x_test.shape} with type {x_test.dtype}")
    else:
        logging.info(f"VAE full train mode without test-samples")
        logging.info(f"VAE train data shape {x_train.shape} with type {x_train.dtype}")

    (vae, encoder) = define_vae_model(opts, output_bias=initial_bias)
    # Train the VAE on the training data
    callback_list = callbacks.autoencoder_callback(
        checkpoint_path=f"{save_path}.h5", opts=opts
    )

    vae_hist = vae.fit(
        x_train,
        x_train,
        epochs=opts.aeEpochs,
        batch_size=opts.aeBatchSize,
        verbose=opts.verbose,
        callbacks=[callback_list],
        validation_data=(x_test, x_test) if opts.testSize > 0.0 else None,
    )

    # Save the VAE weights
    ht.store_and_plot_history(
        base_file_name=save_path,
        hist=vae_hist,
    )
    # Re-define autoencoder and encoder using your function
    callback_autoencoder, callback_encoder = define_vae_model(opts)
    callback_autoencoder.load_weights(filepath=f"{save_path}.h5")

    for i, layer in enumerate(callback_encoder.layers):
        layer.set_weights(callback_autoencoder.layers[i].get_weights())

    # Save the encoder model
    encoder_save_path = f"{save_path}_encoder.h5"
    callback_encoder.save_weights(filepath=encoder_save_path)

    return encoder, train_indices, test_indices
