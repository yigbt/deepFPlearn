import os.path
from os.path import basename
import math

import keras.metrics
import numpy as np
import pandas as pd
import logging

from keras.models import Model
from keras.layers import Input, Dense
from tensorflow.keras import optimizers
import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split

from dfpl import options
from dfpl import callbacks
from dfpl import history as ht
from dfpl import settings


def define_ac_model(opts: options.Options,
                    my_loss: str = "binary_crossentropy") -> (Model, Model):
    """
    This function provides an autoencoder model to reduce a certain input to a compressed version.

    :param opts: Training options that provide values for adjusting the neural net
    :param encoding_dim: Size of the compressed representation. Default: 85
    :param input_size: Size of the input. Default: 2048
    :param my_loss: Loss function, see Keras Loss functions for potential values. Default: binary_crossentropy
    :param my_lr:
    :param my_decay:
    :return: a tuple of autoencoder and encoder models
    """
    input_size = opts.fpSize
    encoding_dim = opts.encFPSize
    ac_optimizer = optimizers.Adam(learning_rate=opts.aeLearningRate,
                                   decay=opts.aeLearningRateDecay)

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
        # encoding layers, incl. bottle neck
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
        decoded = Dense(units=input_size,
                        activation='sigmoid')(decoded)

    else:
        # output layer
        decoded = Dense(units=input_size,
                        activation='sigmoid')(encoded)

    autoencoder = Model(input_vec, decoded)
    encoder = Model(input_vec, encoded)

    autoencoder.summary(print_fn=logging.info)
    encoder.summary(print_fn=logging.info)

    # We compile the autoencoder model with adam optimizer.
    # As fingerprint positions have a value of 0 or 1 we use binary_crossentropy as the loss function
    autoencoder.compile(optimizer=ac_optimizer,
                        loss=my_loss,
                        metrics=[tfa.metrics.F1Score(num_classes=1, threshold=0.5, average="micro"), keras.metrics.Precision(), keras.metrics.Recall()]
                        )

    return autoencoder, encoder


def train_full_ac(df: pd.DataFrame, opts: options.Options) -> Model:
    """
    Train an autoencoder on the given feature matrix X. Response matrix is only used to
    split meaningfully in test and train data set.

    :param opts: Command line arguments as defined in options.py
    :param df: Pandas dataframe that contains the smiles/inchi data for training the autoencoder
    :return: The encoder model of the trained autoencoder
    """

    # Set up the model of the AC w.r.t. the input size and the dimension of the bottle neck (z!)
    (autoencoder, encoder) = define_ac_model(opts)

    # define output file for autoencoder and encoder weights
    if opts.ecWeightsFile == "":
        logging.info("No AE encoder weights file specified")
        base_file_name = os.path.splitext(basename(opts.inputFile))[0]
        logging.info(f"(auto)encoder weights will be saved in {base_file_name}.[auto]encoder.hdf5")
        ac_weights_file = os.path.join(opts.outputDir, base_file_name + ".autoencoder.weights.hdf5")
        ec_weights_file = os.path.join(opts.outputDir, base_file_name + ".encoder.weights.hdf5")
    else:
        logging.info(f"AE encoder will be saved in {opts.ecWeightsFile}")
        base_file_name = os.path.splitext(basename(opts.ecWeightsFile))[0]
        ac_weights_file = os.path.join(opts.outputDir, base_file_name + ".autoencoder.weights.hdf5")
        ec_weights_file = os.path.join(opts.outputDir, opts.ecWeightsFile)

    # collect the callbacks for training
    callback_list = callbacks.autoencoder_callback(checkpoint_path=ac_weights_file, opts=opts)

    # Select all fps that are valid and turn them into a numpy array
    # This step is crucial for speed!!!
    fp_matrix = np.array(df[df["fp"].notnull()]["fp"].to_list(),
                         dtype=settings.ac_fp_numpy_type,
                         copy=settings.numpy_copy_values)
    logging.info(f"Training AC on a matrix of shape {fp_matrix.shape} with type {fp_matrix.dtype}")

    # split data into test and training data
    x_train, x_test = train_test_split(fp_matrix,
                                       test_size=0.2,
                                       random_state=42)
    logging.info(f"AC train data shape {x_train.shape} with type {x_train.dtype}")
    logging.info(f"AC test data shape {x_test.shape} with type {x_test.dtype}")

    auto_hist = autoencoder.fit(x_train, x_train,
                                callbacks=callback_list,
                                epochs=opts.aeEpochs,
                                batch_size=256,
                                verbose=opts.verbose,
                                validation_data=(x_test, x_test))
    logging.info(f"Autoencoder weights stored in file: {ac_weights_file}")

    ht.store_and_plot_history(base_file_name=os.path.join(opts.outputDir, base_file_name + ".AC"),
                              hist=auto_hist)

    # encoder.save_weights(ec_weights_file) # these are the wrong weights! we need those from the callback model
    # logging.info(f"Encoder weights stored in file: {ec_weights_file}")
    # save AE callback model
    (callback_autoencoder, callback_encoder) = define_ac_model(opts)
    callback_autoencoder.load_weights(filepath=ac_weights_file)
    callback_encoder.save(filepath=opts.ecModelDir)

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
