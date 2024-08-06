import logging
import math
import os.path
from os.path import basename
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import wandb
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers, losses, optimizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from dfpl import callbacks
from dfpl import history as ht
from dfpl import settings
from dfpl.train import TrainOptions
from dfpl.utils import ae_scaffold_split, weight_split


def define_ac_model(opts: TrainOptions, output_bias=None) -> Tuple[Model, Model]:
    """
    This function provides an autoencoder model to reduce a certain input to a compressed version.

    :param opts: Training options that provide values for adjusting the neural net
    :param output_bias: Bias used to initialize the last layer. It gives the net a head start in training on
    imbalanced data (which the fingerprints are, because they have many more 0's than 1's in them).
    :return: a tuple of autoencoder and encoder models
    """
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

        # 1st decoding layer
        factor_units = 2 ** (hidden_layer_count - 1)
        if opts.aeActivationFunction != "selu":
            decoded = Dense(
                units=int(input_size / factor_units),
                activation=opts.aeActivationFunction,
            )(encoded)
        else:
            decoded = Dense(
                units=int(input_size / factor_units),
                activation=opts.aeActivationFunction,
                kernel_initializer="lecun_normal",
            )(encoded)

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
        # to either 0 or 1 and hence we use sigmoid activation function.
        decoded = Dense(
            units=input_size, activation="sigmoid", bias_initializer=output_bias
        )(decoded)

    else:
        # output layer
        decoded = Dense(
            units=input_size, activation="sigmoid", bias_initializer=output_bias
        )(encoded)

    autoencoder = Model(input_vec, decoded)
    encoder = Model(input_vec, encoded)
    autoencoder.summary(print_fn=logging.info)

    autoencoder.compile(
        optimizer=ac_optimizer,
        loss=losses.BinaryCrossentropy(),
        # metrics=[
        #     metrics.AUC(),
        #     metrics.Precision(),
        #     metrics.Recall()
        # ]
    )
    return autoencoder, encoder


def train_full_ac(df: pd.DataFrame, opts: TrainOptions) -> Model:
    """
    Trains an autoencoder on the given feature matrix X. The response matrix is only used to
    split the data into meaningful test and train sets.

    :param opts: Command line arguments
    :param df: Pandas dataframe that contains the SMILES/InChI data for training the autoencoder
    :return: The encoder model of the trained autoencoder
    """

    # If wandb tracking is enabled for autoencoder weights but not for the main program, initialize a new wandb run
    if opts.aeWabTracking and not opts.wabTracking:
        wandb.init(project=f"AE_{opts.aeSplitType}")

    # Define output files for autoencoder and encoder weights
    if opts.ecWeightsFile == "":
        # If no encoder weights file is specified, use the input file name to generate a default file name
        logging.info("No AE encoder weights file specified")
        base_file_name = (
            os.path.splitext(basename(opts.inputFile))[0] + opts.aeSplitType
        )
        logging.info(
            f"(auto)encoder weights will be saved in {base_file_name}.autoencoder.hdf5"
        )
        ac_weights_file = os.path.join(
            opts.outputDir, base_file_name + ".autoencoder.weights.hdf5"
        )
        # ec_weights_file = os.path.join(
        #     opts.outputDir, base_file_name + ".encoder.weights.hdf5"
        # )
    else:
        # If an encoder weights file is specified, use it as the encoder weights file name
        logging.info(f"AE encoder will be saved in {opts.ecWeightsFile}")
        base_file_name = (
            os.path.splitext(basename(opts.ecWeightsFile))[0] + opts.aeSplitType
        )
        ac_weights_file = os.path.join(
            opts.outputDir, base_file_name + ".autoencoder.weights.hdf5"
        )
        # ec_weights_file = os.path.join(opts.outputDir, opts.ecWeightsFile)

    # Collect the callbacks for training
    callback_list = callbacks.autoencoder_callback(
        checkpoint_path=ac_weights_file, opts=opts
    )

    # Select all fingerprints that are valid and turn them into a numpy array
    fp_matrix = np.array(
        df[df["fp"].notnull()]["fp"].to_list(),
        dtype=settings.ac_fp_numpy_type,
        copy=settings.numpy_copy_values,
    )
    logging.info(
        f"Training AC on a matrix of shape {fp_matrix.shape} with type {fp_matrix.dtype}"
    )

    # When training the final AE, we don't want any test data. We want to train it on all available fingerprints.
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

    # Check if we're doing training/testing mode or full training mode
    if opts.testSize > 0.0:
        logging.info(f"AE training/testing mode with train- and test-samples")
        logging.info(f"AC train data shape {x_train.shape} with type {x_train.dtype}")
        logging.info(f"AC test data shape {x_test.shape} with type {x_test.dtype}")
    else:
        logging.info(f"AE full train mode without test-samples")
        logging.info(f"AC train data shape {x_train.shape} with type {x_train.dtype}")

    # Set up the model of the AC w.r.t. the input size and the dimension of the bottle neck (z!)
    (autoencoder, encoder) = define_ac_model(opts, output_bias=initial_bias)

    # Train the autoencoder on the training data
    auto_hist = autoencoder.fit(
        x_train,
        x_train,
        callbacks=callback_list,
        epochs=opts.aeEpochs,
        batch_size=opts.aeBatchSize,
        verbose=opts.verbose,
        validation_data=(x_test, x_test) if opts.testSize > 0.0 else None,
    )
    logging.info(f"Autoencoder weights stored in file: {ac_weights_file}")

    # Store the autoencoder training history and plot the metrics
    ht.store_and_plot_history(
        base_file_name=os.path.join(opts.outputDir, base_file_name + ".AC"),
        hist=auto_hist,
    )

    # Save the autoencoder callback model to disk
    save_path = os.path.join(opts.ecModelDir, f"{opts.aeSplitType}_autoencoder")
    if opts.testSize > 0.0:
        (callback_autoencoder, callback_encoder) = define_ac_model(opts)
        callback_encoder.save(filepath=save_path)
    else:
        encoder.save(filepath=save_path)
    # Return the encoder model of the trained autoencoder
    return encoder, train_indices, test_indices


def compress_fingerprints(dataframe: pd.DataFrame, encoder: Model) -> pd.DataFrame:
    """
    Adds a column of the compressed version of the fingerprints to the original dataframe.

    :param dataframe: Dataframe containing a column named 'fp' with the fingerprints
    :param encoder: The trained autoencoder that is used for compressing the fingerprints
    :return: The input dataframe extended by a column containing the compressed version of the fingerprints
    """
    logging.info("Adding compressed fingerprints")
    idx = dataframe[dataframe["fp"].notnull()].index
    fp_matrix = np.array(
        dataframe[dataframe["fp"].notnull()]["fp"].to_list(),
        dtype=settings.ac_fp_numpy_type,
        copy=settings.numpy_copy_values,
    )
    logging.info(
        f"Using input matrix of shape {fp_matrix.shape} with type {fp_matrix.dtype}"
    )
    logging.info("Compressed fingerprints are added to input dataframe.")
    dataframe["fpcompressed"] = pd.DataFrame(
        {"fpcompressed": [s for s in encoder.predict(fp_matrix)]}, idx
    )
    return dataframe


def visualize_fingerprints(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    save_as: str,
):
    # Calculate the number of samples to be taken from each set
    num_samples = 1000
    train_samples = int(num_samples * len(train_indices) / len(df))
    test_samples = num_samples - train_samples

    # Assign train and test data points separately
    train_data = df.loc[train_indices]
    test_data = df.loc[test_indices]

    # Sample train and test data points
    train_data_sampled = train_data.sample(n=train_samples, random_state=42)
    test_data_sampled = test_data.sample(n=test_samples, random_state=42)

    # Concatenate the sampled train and test data
    df_sampled = pd.concat([train_data_sampled, test_data_sampled])

    # Convert the boolean values in the after_col column to floats
    df_sampled[after_col] = df_sampled[after_col].apply(
        lambda x: np.array(x, dtype=float)
    )

    df_sampled.loc[train_data_sampled.index, "set"] = "train"
    df_sampled.loc[test_data_sampled.index, "set"] = "test"
    # Apply UMAP
    umap_model = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42
    )
    # Filter out the rows with invalid arrays
    umap_results = umap_model.fit_transform(df_sampled[after_col].tolist())
    # Add UMAP results to the DataFrame
    df_sampled["umap_x"] = umap_results[:, 0]
    df_sampled["umap_y"] = umap_results[:, 1]

    # Define custom color palette
    palette = {"train": "blue", "test": "red"}

    # Create the scatter plot
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(10, 8))
    split = save_as.split("_", 1)
    part_after_underscore = split[1]
    split_type = part_after_underscore.split(".")[0]
    # Plot the UMAP results
    for label, grp in df_sampled.groupby("set"):
        set_label = label
        color = palette[set_label]
        alpha = (
            0.09 if set_label == "train" else 0.9
        )  # Set different opacities for train and test
        ax.scatter(
            grp["umap_x"], grp["umap_y"], label=f"{set_label}", c=color, alpha=alpha
        )

    # Customize the plot
    ax.set_title(
        f"UMAP visualization of molecular fingerprints using {split_type} split",
        fontsize=14,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title="", loc="upper right")
    sns.despine(ax=ax, offset=10)
    save_path = os.path.join(os.getcwd(), save_as)
    plt.savefig(save_path)
