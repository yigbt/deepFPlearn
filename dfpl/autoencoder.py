import logging
import math
import os.path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
import wandb
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from dfpl import callbacks
from dfpl import history as ht
from dfpl import options, settings
from dfpl.utils import ae_scaffold_split, weight_split


def create_dense_layer(inputs, units, activation):
    """Create a Dense layer with optional SELU initialization."""
    return Dense(
        units=units,
        activation=activation,
        kernel_initializer="lecun_normal" if activation == "selu" else 'glorot_uniform'
    )(inputs)


def define_ac_model(opts: options.Options) -> Tuple[Model, Model]:
    """
    Define and compile an autoencoder model with the specified encoding dimension, with a mirrored decoder.

    :param opts: Training options containing model parameters.
    :param output_bias: Bias for the output layer, used for initializing the last layer.
    :return: Tuple containing the autoencoder and encoder models.
    """
    input_size = opts.fpSize
    encoding_dim = opts.encFPSize  # Desired encoding dimension
    lr_schedule = optimizers.schedules.ExponentialDecay(
        opts.aeLearningRate,
        decay_steps=math.ceil(7000 / opts.batchSize) * 3,
        decay_rate=opts.aeLearningRateDecay,
        staircase=True,
    )
    ac_optimizer = optimizers.legacy.Adam(learning_rate=lr_schedule)

    input_vec = Input(shape=(input_size,))
    initial_layer_size = int(input_size / 2)
    encoded = create_dense_layer(input_vec, initial_layer_size, opts.aeActivationFunction)

    # Start `layer_sizes` with the initial layer size (1024)
    layer_sizes = [initial_layer_size]

    # Build intermediate encoding layers and store their sizes
    hidden_layer_count = round(math.log2(input_size / encoding_dim))
    for i in range(1, hidden_layer_count):
        factor_units = 2 ** (i + 1)
        layer_size = int(input_size / factor_units)
        layer_sizes.append(layer_size)
        encoded = create_dense_layer(encoded, layer_size, opts.aeActivationFunction)

    # Build decoder layers in exact reverse order, excluding the first layer size
    decoded = encoded
    for layer_size in reversed(layer_sizes[:-1]):
        decoded = Dense(units=layer_size, activation=opts.aeActivationFunction)(decoded)

    # Final output layer to reconstruct input
    decoded = Dense(units=input_size, activation="sigmoid")(decoded)

    # Define autoencoder and encoder models
    autoencoder = Model(input_vec, decoded)
    encoder = Model(input_vec, encoded)

    # Compile autoencoder
    autoencoder.compile(optimizer=ac_optimizer, loss=losses.BinaryCrossentropy())
    autoencoder.summary(print_fn=logging.info)

    return autoencoder, encoder



def setup_train_test_split(df: pd.DataFrame, opts: options.Options) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sets up the training and test split based on the provided options.

    :param opts: Command line arguments as defined in options.py
    :param df: Pandas dataframe that contains the SMILES/InChI data for training the autoencoder
    :return: Tuple containing training data, test data, training indices, test indices, and initial bias.
    """

    # Select all fingerprints that are valid and turn them into a numpy array
    fp_matrix = np.array(
        df[df["fp"].notnull()]["fp"].to_list(),
        dtype=settings.ac_fp_numpy_type,
        copy=settings.numpy_copy_values,
    )

    logging.info(f"Setting up train/test split on a matrix of shape {fp_matrix.shape} with type {fp_matrix.dtype}")

    # Validate test size
    assert 0.0 <= opts.testSize <= 0.5
    initial_indices = np.arange(fp_matrix.shape[0])

    if opts.aeSplitType == "random":
        logging.info("Using random split for training.")
        if opts.testSize > 0.0:
            x_train, x_test, train_indices, test_indices = train_test_split(
                fp_matrix, initial_indices, test_size=opts.testSize, random_state=42
            )
        else:
            x_train = fp_matrix
            x_test = None
            train_indices = initial_indices
            test_indices = None

    elif opts.aeSplitType == "scaffold_balanced":
        logging.info("Using scaffold split for training.")
        train_data, val_data, test_data = ae_scaffold_split(
            df, sizes=(1 - opts.testSize, 0.0, opts.testSize), balanced=True, seed=42
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
        train_indices = df[df.index.isin(train_data[train_data["fp"].notnull()].index)].index.to_numpy()
        test_indices = df[df.index.isin(test_data[test_data["fp"].notnull()].index)].index.to_numpy()

    elif opts.aeSplitType == "molecular_weight":
        logging.info("Using molecular weight split for training.")
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
        sorted_indices = df_sorted.index.to_numpy()
        train_indices = sorted_indices[df.index.isin(train_data.index)]
        test_indices = sorted_indices[df.index.isin(test_data.index)]

    else:
        raise ValueError(f"Invalid split type: {opts.aeSplitType}")

    return x_train, x_test, train_indices, test_indices


def train_full_ac(df: pd.DataFrame, opts: options.Options) -> Model:
    """
    Trains an autoencoder on the given feature matrix X. The response matrix is only used to
    split the data into meaningful test and train sets.

    :param opts: Command line arguments as defined in options.py
    :param df: Pandas dataframe that contains the SMILES/InChI data for training the autoencoder
    :return: The encoder model of the trained autoencoder
    """

    # If wandb tracking is enabled for autoencoder weights but not for the main program, initialize a new wandb run
    if opts.aeWabTracking:
        wandb.init(project=f"AE_{opts.aeSplitType}")

    save_path = os.path.join(opts.ecModelDir, f"autoencoder_weights.h5")

    # Set up train/test split
    x_train, x_test, train_indices, test_indices = setup_train_test_split(df, opts)

    # Log training mode
    if opts.testSize > 0.0:
        logging.info(f"AE training/testing mode with train- and test-samples")
        logging.info(f"AC train data shape {x_train.shape} with type {x_train.dtype}")
        logging.info(f"AC test data shape {x_test.shape} with type {x_test.dtype}")
    else:
        logging.info(f"AE full train mode without test-samples")
        logging.info(f"AC train data shape {x_train.shape} with type {x_train.dtype}")

    # Set up the model of the AC
    (autoencoder, encoder) = define_ac_model(opts)
    callback_list = callbacks.autoencoder_callback(checkpoint_path=save_path, opts=opts)

    # Train the autoencoder on the training data
    auto_hist = autoencoder.fit(
        x_train,
        x_train,
        callbacks=[callback_list],
        epochs=opts.aeEpochs,
        batch_size=opts.aeBatchSize,
        verbose=opts.verbose,
        validation_data=(x_test, x_test) if opts.testSize > 0.0 else None,
    )

    # Store the autoencoder training history and plot the metrics
    ht.store_and_plot_history(
        base_file_name=save_path,
        hist=auto_hist,
    )
    # load the model with the best weights
    autoencoder.load_weights(save_path)
    # Save the encoder weights
    encoder.save_weights(os.path.join(opts.ecModelDir, "encoder_weights.h5"))


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
    comressed_col: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    save_as: str,
):
    # Calculate the number of samples to be taken from each set
    num_samples = 10
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
    df_sampled[comressed_col] = df_sampled[comressed_col].apply(
        lambda x: np.array(x, dtype=float)
    )

    df_sampled.loc[train_data_sampled.index, "set"] = "train"
    df_sampled.loc[test_data_sampled.index, "set"] = "test"
    # Apply UMAP
    umap_model = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42
    )
    # Filter out the rows with invalid arrays
    umap_results = umap_model.fit_transform(df_sampled[comressed_col].tolist())
    # Add UMAP results to the DataFrame
    df_sampled["umap_x"] = umap_results[:, 0]
    df_sampled["umap_y"] = umap_results[:, 1]

    # Define custom color palette
    palette = {"train": "blue", "test": "red"}

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
