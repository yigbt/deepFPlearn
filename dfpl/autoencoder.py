import os.path
from os.path import basename
import math
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from wandb.keras import WandbCallback
from umap import UMAP
from sklearn.cluster import KMeans
from dfpl.utils import *
import numpy as np
import pandas as pd
import logging
import wandb
import tensorflow.keras.metrics as metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers, losses, initializers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from dfpl import options
from dfpl import callbacks
from dfpl import history as ht
from dfpl import settings


def define_ac_model(opts: options.Options, output_bias=None) -> (Model, Model):
    """
    This function provides an autoencoder model to reduce a certain input to a compressed version.

    :param opts: Training options that provide values for adjusting the neural net
    :param output_bias: Bias used to initialize the last layer. It gives the net a head start in training on
    imbalanced data (which the fingerprints are, because they have many more 0's than 1's in them).
    :return: a tuple of autoencoder and encoder models
    """
    input_size = opts.fpSize
    encoding_dim = opts.encFPSize
    ac_optimizer = optimizers.Adam(learning_rate=opts.aeLearningRate,
                                   decay=opts.aeLearningRateDecay)

    if output_bias is not None:
        output_bias = initializers.Constant(output_bias)

    # get the number of meaningful hidden layers (latent space included)
    hidden_layer_count = round(math.log2(input_size / encoding_dim))

    # the input placeholder
    input_vec = Input(shape=(input_size,))

    # 1st hidden layer, that receives weights from input layer
    # equals bottleneck layer, if hidden_layer_count==1!
    if opts.aeActivationFunction != "selu":
        encoded = Dense(units=int(input_size / 2), activation=opts.aeActivationFunction)(input_vec)
    else:
        encoded = Dense(units=int(input_size / 2),
                        activation=opts.aeActivationFunction,
                        kernel_initializer="lecun_normal")(input_vec)

    if hidden_layer_count > 1:
        # encoding layers, incl. bottle-neck
        for i in range(1, hidden_layer_count):
            factor_units = 2 ** (i + 1)
            # print(f'{factor_units}: {int(input_size / factor_units)}')
            if opts.aeActivationFunction != "selu":
                encoded = Dense(units=int(input_size / factor_units), activation=opts.aeActivationFunction)(encoded)
            else:
                encoded = Dense(units=int(input_size / factor_units),
                                activation=opts.aeActivationFunction,
                                kernel_initializer="lecun_normal")(encoded)

        # 1st decoding layer
        factor_units = 2 ** (hidden_layer_count - 1)
        if opts.aeActivationFunction != "selu":
            decoded = Dense(units=int(input_size / factor_units), activation=opts.aeActivationFunction)(encoded)
        else:
            decoded = Dense(units=int(input_size / factor_units),
                            activation=opts.aeActivationFunction,
                            kernel_initializer="lecun_normal")(encoded)

        # decoding layers
        for i in range(hidden_layer_count - 2, 0, -1):
            factor_units = 2 ** i
            # print(f'{factor_units}: {int(input_size/factor_units)}')
            if opts.aeActivationFunction != "selu":
                decoded = Dense(units=int(input_size / factor_units), activation=opts.aeActivationFunction)(decoded)
            else:
                decoded = Dense(units=int(input_size / factor_units),
                                activation=opts.aeActivationFunction,
                                kernel_initializer="lecun_normal")(decoded)

        # output layer
        # The output layer needs to predict the probability of an output which needs
        # to either 0 or 1 and hence we use sigmoid activation function.
        decoded = Dense(units=input_size, activation='sigmoid', bias_initializer=output_bias)(decoded)

    else:
        # output layer
        decoded = Dense(units=input_size, activation='sigmoid', bias_initializer=output_bias)(encoded)

    autoencoder = Model(input_vec, decoded)
    encoder = Model(input_vec, encoded)
    autoencoder.summary(print_fn=logging.info)

    autoencoder.compile(optimizer=ac_optimizer,
                        loss=losses.BinaryCrossentropy(),
                        metrics=[
                            metrics.AUC(),
                            metrics.Precision(),
                            metrics.Recall()
                        ]
                        )
    return autoencoder, encoder


def train_full_ac(df: pd.DataFrame, opts: options.Options) -> Model:
    """
    Trains an autoencoder on the given feature matrix X. The response matrix is only used to
    split the data into meaningful test and train sets.

    :param opts: Command line arguments as defined in options.py
    :param df: Pandas dataframe that contains the SMILES/InChI data for training the autoencoder
    :return: The encoder model of the trained autoencoder
    """

    # If wandb tracking is enabled for autoencoder weights but not for the main program, initialize a new wandb run
    if opts.aeWabTracking and  not opts.wabTracking:
        wandb.init(project=f"AE_{opts.aeSplitType}", config=opts)

    # Define output files for autoencoder and encoder weights
    if opts.ecWeightsFile == "":
        # If no encoder weights file is specified, use the input file name to generate a default file name
        logging.info("No AE encoder weights file specified")
        base_file_name = os.path.splitext(basename(opts.inputFile))[0] + opts.aeSplitType
        logging.info(f"(auto)encoder weights will be saved in {base_file_name}.autoencoder.hdf5")
        ac_weights_file = os.path.join(opts.outputDir, base_file_name + ".autoencoder.weights.hdf5")
        ec_weights_file = os.path.join(opts.outputDir, base_file_name + ".encoder.weights.hdf5")
    else:
        # If an encoder weights file is specified, use it as the encoder weights file name
        logging.info(f"AE encoder will be saved in {opts.ecWeightsFile}")
        base_file_name = os.path.splitext(basename(opts.ecWeightsFile))[0] + opts.aeSplitType
        ac_weights_file = os.path.join(opts.outputDir, base_file_name + ".autoencoder.weights.hdf5")
        ec_weights_file = os.path.join(opts.outputDir, opts.ecWeightsFile)

    # Collect the callbacks for training
    callback_list = callbacks.autoencoder_callback(checkpoint_path=ac_weights_file, opts=opts)

    # Select all fingerprints that are valid and turn them into a numpy array
    fp_matrix = np.array(df[df["fp"].notnull()]["fp"].to_list(),
                         dtype=settings.ac_fp_numpy_type,
                         copy=settings.numpy_copy_values)
    logging.info(f"Training AC on a matrix of shape {fp_matrix.shape} with type {fp_matrix.dtype}")

    # When training the final AE, we don't want any test data. We want to train it on all available fingerprints.
    assert (0.0 <= opts.testSize <= 0.5)
    if opts.aeSplitType == "random":
        logging.info("Training autoencoder using random split")
        if opts.testSize > 0.0:
            # Split data into test and training data
            if opts.aeWabTracking:
                x_train, x_test = train_test_split(fp_matrix, test_size=opts.testSize, random_state=42)
            else:
                x_train, x_test = train_test_split(fp_matrix, test_size=opts.testSize)
        else:
            x_train = fp_matrix
            x_test = None
    # Split the data into train and test sets using scaffold split
    elif opts.aeSplitType == "scaffold_balanced":
        logging.info("Training autoencoder using scaffold split")
        if opts.testSize > 0.0:
            if opts.aeWabTracking:
                train_data, val_data, test_data = scaffold_split(df, sizes=(1-opts.testSize,0.0, opts.testSize), balanced=True,seed=42)
            else:
                train_data, val_data, test_data = scaffold_split(df, sizes=(1-opts.testSize,0.0, opts.testSize), balanced=True)
            x_train = np.array(train_data[train_data["fp"].notnull()]["fp"].to_list(),
                             dtype=settings.ac_fp_numpy_type,
                             copy=settings.numpy_copy_values)
            x_test = np.array(test_data[test_data["fp"].notnull()]["fp"].to_list(),
                           dtype=settings.ac_fp_numpy_type,
                           copy=settings.numpy_copy_values)
            x_val = None
        else:
            x_test = None
            x_val = None
            x_train = fp_matrix

    else:
        raise ValueError(f"Invalid split type: {opts.split_type}")


    # Calculate the initial bias aka the log ratio between 1's and 0'1 in all fingerprints
    ids, counts = np.unique(x_train.flatten(), return_counts=True)
    count_dict = dict(zip(ids, counts))
    if count_dict[0] == 0:
        initial_bias = None
        logging.info("No zeroes in training labels. Setting initial_bias to None.")
    else:
        initial_bias = np.log([count_dict[1]/count_dict[0]])
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
    auto_hist = autoencoder.fit(x_train, x_train,
                                callbacks=callback_list,
                                epochs=opts.aeEpochs,
                                batch_size=opts.aeBatchSize,
                                verbose=opts.verbose,
                                validation_data=(x_test, x_test) if opts.testSize > 0.0 else None
                                )
    logging.info(f"Autoencoder weights stored in file: {ac_weights_file}")

    # Log the autoencoder training metrics to W&B if enabled
    if opts.aeWabTracking and not opts.wabTracking:
        wandb.log({
            "AE_loss": auto_hist.history['loss'][-1],
            "AE_val_loss": auto_hist.history['val_loss'][-1],
            "AE_auc": auto_hist.history['auc'][-1],
            "AE_val_auc": auto_hist.history['val_auc'][-1],
            "AE_precision": auto_hist.history['precision'][-1],
            "AE_val_precision": auto_hist.history['val_precision'][-1],
            "AE_recall": auto_hist.history['recall'][-1],
            "AE_val_recall": auto_hist.history['val_recall'][-1],
            "num_epochs": len(auto_hist.history['loss'])
        })

    # Store the autoencoder training history and plot the metrics
    ht.store_and_plot_history(base_file_name=os.path.join(opts.outputDir, base_file_name + ".AC"),
                              hist=auto_hist)

    # Save the autoencoder callback model to disk
    save_path = os.path.join(opts.ecModelDir, f"{opts.aeSplitType}_autoencoder.h5")
    if opts.testSize > 0.0:
        (callback_autoencoder, callback_encoder) = define_ac_model(opts)
        callback_encoder.save(filepath=save_path)
    else:
        encoder.save(filepath=save_path)

    # Return the encoder model of the trained autoencoder
    return encoder



def compress_fingerprints(dataframe: pd.DataFrame,
                          encoder: Model) -> pd.DataFrame:
    """
    Adds a column of the compressed version of the fingerprints to the original dataframe.

    :param dataframe: Dataframe containing a column named 'fp' with the fingerprints
    :param encoder: The trained autoencoder that is used for compressing the fingerprints
    :return: The input dataframe extended by a column containing the compressed version of the fingerprints
    """
    logging.info("Adding compressed fingerprints")
    idx = dataframe[dataframe["fp"].notnull()].index
    fp_matrix = np.array(dataframe[dataframe["fp"].notnull()]["fp"].to_list(),
                         dtype=settings.ac_fp_numpy_type,
                         copy=settings.numpy_copy_values)
    logging.info(f"Using input matrix of shape {fp_matrix.shape} with type {fp_matrix.dtype}")
    logging.info("Compressed fingerprints are added to input dataframe.")
    dataframe['fpcompressed'] = pd.DataFrame({'fpcompressed': [s for s in encoder.predict(fp_matrix)]}, idx)

    return dataframe

def visualize_fingerprints(df: pd.DataFrame, before_col: str, after_col: str, output_dir: str, opts:options.Options):    
    # Convert the boolean values in the before_col column to floats
    df[before_col] = df[before_col].apply(lambda x: np.array(x, dtype=float))

    # Get the fingerprints before and after compression
    before_fingerprints = np.array(df[before_col].to_list())
    after_fingerprints = np.array(df[after_col].to_list())
    
    # Create UMAP and t-SNE embeddings for the fingerprints before and after compression
    before_umap_embedding = UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42).fit_transform(before_fingerprints)
    after_umap_embedding = UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42).fit_transform(after_fingerprints)
    # before_tsne_embedding = TSNE().fit_transform(before_fingerprints)
    # after_tsne_embedding = TSNE().fit_transform(after_fingerprints)

    # Apply the elbow method to find the best number of clusters for k-means
    # wcss = []
    # max_clusters = 10
    # for k in range(1, max_clusters):
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(before_umap_embedding)
    #     wcss.append(kmeans.inertia_)
    # plt.plot(range(1, max_clusters), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.savefig(os.path.join(output_dir, f'wcss_plot_{opts.split_type}.png'))
    # plt.show()
    # 3 or 4 clusters seem like a good choice

    # Use silhouette score to choose number of clusters
    # max_clusters = 5
    # before_umap_scores = []
    # for n_clusters in range(2, max_clusters+1):
    #     labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(before_umap_embedding)
    #     score = silhouette_score(before_umap_embedding, labels)
    #     before_umap_scores.append(score)
    # optimal_n_clusters = np.argmax(before_umap_scores) + 2
    # print(f"Optimal number of clusters for before (UMAP) is {optimal_n_clusters}")

    # Apply k-means clustering to the embeddings
    n_clusters = 4 # optimal_n_clusters= 3 based on silhouette score but 4 was previously used
    before_umap_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(before_umap_embedding)
    after_umap_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(after_umap_embedding)
    # before_tsne_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(before_tsne_embedding)
    # after_tsne_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(after_tsne_embedding)
    # Assign same colormap to all plots
    cmap = plt.get_cmap('rainbow', n_clusters)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    fig.suptitle('Dimensionality reduction and clustering of fingerprints')
    before_umap_scatter = axes[0].scatter(before_umap_embedding[:, 0], before_umap_embedding[:, 1], c=before_umap_labels, s=5, cmap=cmap)
    axes[0].set_title('Before compression (UMAP)')
    after_umap_scatter = axes[1].scatter(after_umap_embedding[:, 0], after_umap_embedding[:, 1], c=after_umap_labels, s=5, cmap=cmap)
    axes[1].set_title('After compression (UMAP)')
    # before_tsne_scatter = axes[1, 0].scatter(before_tsne_embedding[:, 0], before_tsne_embedding[:, 1], c=before_tsne_labels, s=5, cmap=cmap)
    # axes[1, 0].set_title('Before compression (t-SNE)')
    # after_tsne_scatter = axes[1, 1].scatter(after_tsne_embedding[:, 0], after_tsne_embedding[:, 1], c=after_tsne_labels, s=5, cmap=cmap)
    # axes[1, 1].set_title('After compression (t-SNE)')
    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fingerprints_{opts.split_type}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
