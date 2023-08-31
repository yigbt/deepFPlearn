import csv
import logging
import math
import os.path
from os.path import basename
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

from dfpl import callbacks
from dfpl import history as ht
from dfpl import options, settings
from dfpl.utils import ae_scaffold_split, weight_split

disable_eager_execution()


def define_vae_model(opts: options.Options, output_bias=None) -> Tuple[Model, Model]:
    input_size = opts.fpSize
    encoding_dim = opts.encFPSize
    ac_optimizer = optimizers.Adam(
        learning_rate=opts.aeLearningRate, decay=opts.aeLearningRateDecay
    )

    if output_bias is not None:
        output_bias = initializers.Constant(output_bias)

    # get the number of meaningful hidden layers (latent space included)
    hidden_layer_count = round(math.log2(input_size / encoding_dim))

    # the input placeholder
    input_vec = Input(shape=(input_size,))

    # 1st hidden layer, that receives weights from input layer
    # equals bottleneck layer, if hidden_layer_count==1!
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

    if hidden_layer_count > 1:
        # encoding layers, incl. bottle-neck
        for i in range(1, hidden_layer_count):
            factor_units = 2 ** (i + 1)
            # print(f'{factor_units}: {int(input_size / factor_units)}')
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
        factor_units = 2 ** (hidden_layer_count - 1)
        if opts.aeActivationFunction != "selu":
            z_mean = Dense(
                units=int(input_size / factor_units),
                activation=opts.aeActivationFunction,
            )(encoded)
            z_log_var = Dense(
                units=int(input_size / factor_units),
                activation=opts.aeActivationFunction,
            )(encoded)
        else:
            z_mean = Dense(
                units=int(input_size / factor_units),
                activation=opts.aeActivationFunction,
                kernel_initializer="lecun_normal",
            )(encoded)
            z_log_var = Dense(
                units=int(input_size / factor_units),
                activation=opts.aeActivationFunction,
                kernel_initializer="lecun_normal",
            )(encoded)

        # sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        # sample from latent space
        z = Lambda(sampling, output_shape=(int(input_size / factor_units),))(
            [z_mean, z_log_var]
        )
        decoded = z
        # decoding layers
        for i in range(hidden_layer_count - 2, 0, -1):
            factor_units = 2**i
            # print(f'{factor_units}: {int(input_size/factor_units)}')
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

    else:
        # output layer
        decoded = Dense(
            units=input_size, activation="sigmoid", bias_initializer=output_bias
        )(encoded)

    autoencoder = Model(input_vec, decoded)

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

    # build encoder model
    encoder = Model(input_vec, z_mean)

    return autoencoder, encoder


def train_full_vae(df: pd.DataFrame, opts: options.Options) -> Model:
    """
    Trains an autoencoder on the given feature matrix X. The response matrix is only used to
    split the data into meaningful test and train sets.

    :param opts: Command line arguments as defined in options.py
    :param df: Pandas dataframe that contains the SMILES/InChI data for training the autoencoder
    :return: The encoder model of the trained autoencoder
    """

    # If wandb tracking is enabled for VAE weights but not for the main program, initialize a new wandb run
    if opts.aeWabTracking and not opts.wabTracking:
        wandb.init(project=f"VAE_{opts.aeSplitType}")

    # Define output files for VAE and encoder weights
    if opts.ecWeightsFile == "":
        # If no encoder weights file is specified, use the input file name to generate a default file name
        logging.info("No VAE encoder weights file specified")
        base_file_name = (
            os.path.splitext(basename(opts.inputFile))[0]
            + opts.aeType
            + opts.aeSplitType
        )
        logging.info(
            f"(variational) encoder weights will be saved in {base_file_name}.autoencoder.hdf5"
        )
        vae_weights_file = os.path.join(
            opts.outputDir, base_file_name + ".vae.weights.hdf5"
        )
        # ec_weights_file = os.path.join(
        #     opts.outputDir, base_file_name + ".encoder.weights.hdf5"
        # )
    else:
        # If an encoder weights file is specified, use it as the encoder weights file name
        logging.info(f"VAE encoder will be saved in {opts.ecWeightsFile}")
        base_file_name = (
            os.path.splitext(basename(opts.ecWeightsFile))[0] + opts.aeSplitType
        )
        vae_weights_file = os.path.join(
            opts.outputDir, base_file_name + ".vae.weights.hdf5"
        )
        # ec_weights_file = os.path.join(opts.outputDir, opts.ecWeightsFile)

    # Collect the callbacks for training
    callback_list = callbacks.autoencoder_callback(
        checkpoint_path=vae_weights_file, opts=opts
    )
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
        logging.info("Training VAE using random split")
        train_indices = np.arange(fp_matrix.shape[0])
        if opts.testSize > 0.0:
            # Split data into test and training data
            if opts.aeWabTracking:
                x_train, x_test, _, _ = train_test_split(
                    fp_matrix, train_indices, test_size=opts.testSize, random_state=42
                )
            else:
                x_train, x_test, _, _ = train_test_split(
                    fp_matrix, train_indices, test_size=opts.testSize, random_state=42
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
        else:
            x_train = fp_matrix
            x_test = None
    elif opts.aeSplitType == "molecular_weight":
        logging.info("Training autoencoder using molecular weight split")
        train_indices = np.arange(fp_matrix.shape[0])
        if opts.testSize > 0.0:
            # if opts.aeWabTracking:
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
        else:
            x_train = fp_matrix
            x_test = None
    else:
        raise ValueError(f"Invalid split type: {opts.split_type}")
    if opts.testSize > 0.0:
        train_indices = train_indices[train_indices < x_train.shape[0]]
        test_indices = np.arange(x_train.shape[0], x_train.shape[0] + x_test.shape[0])
    else:
        test_indices = None
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
    vae_hist = vae.fit(
        x_train,
        x_train,
        epochs=opts.aeEpochs,
        batch_size=opts.aeBatchSize,
        verbose=opts.verbose,
        callbacks=callback_list,
        validation_data=(x_test, x_test) if opts.testSize > 0.0 else None,
    )

    # Save the VAE weights
    logging.info(f"VAE weights stored in file: {vae_weights_file}")
    ht.store_and_plot_history(
        base_file_name=os.path.join(opts.outputDir, base_file_name + ".VAE"),
        hist=vae_hist,
    )
    save_path = os.path.join(opts.ecModelDir, f"{opts.aeSplitType}_VAE.h5")
    if opts.testSize > 0.0:
        (callback_vae, callback_encoder) = define_vae_model(opts)
        callback_vae.load_weights(filepath=vae_weights_file)
        callback_encoder.save(filepath=save_path)
    else:
        encoder.save(filepath=save_path)
    latent_space = encoder.predict(fp_matrix)
    latent_space_file = os.path.join(
        opts.outputDir, base_file_name + ".latent_space.csv"
    )
    with open(latent_space_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(latent_space)
    return encoder, train_indices, test_indices
