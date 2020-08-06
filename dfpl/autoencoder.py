import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from keras.models import Model
from keras.layers import Input, Dense
from keras import optimizers

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

from dfpl import options


def autoencoderModel(
        input_size: int = 2048,
        encoding_dim: int = 256,
        myloss: str = "binary_crossentropy",
        mylr: float = 0.001,
        mydecay: float = 0.01) -> (Model, Model):
    """
    This function provides an autoencoder model to reduce a certain input to a compressed version.

    :param encoding_dim: Size of the compressed representation. Default: 85
    :param input_size: Size of the input. Default: 2048
    :param myloss: Loss function, see Keras Loss functions for potential values. Default: binary_crossentropy
    :param mylr:
    :param mydecay:
    :return: a tuple of autoencoder and encoder models
    """

    acOptimizer = optimizers.Adam(learning_rate=mylr, decay=mydecay)  # , beta_1=0.9, beta_2=0.999, amsgrad=False)

    # get the number of meaningful hidden layers (latent space included)
    hiddenLayerCount = round(math.log2(input_size / encoding_dim))

    # the input placeholder
    inputVec = Input(shape=(input_size,))

    # 1st hidden layer, that receives weights from input layer
    # equals bottle neck layer, if hiddenLayerCount==1!
    encoded = Dense(units=int(input_size / 2), activation='relu')(inputVec)

    if hiddenLayerCount > 1:
        # encoding layers, incl. bottle neck
        for i in range(1, hiddenLayerCount):
            factorUnits = 2 ** (i + 1)
            # print(f'{factorUnits}: {int(input_size / factorUnits)}')
            encoded = Dense(units=int(input_size / factorUnits), activation='relu')(encoded)

        #        encoding_dim = int(input_size/factorUnits)

        # 1st decoding layer
        factorUnits = 2 ** (hiddenLayerCount - 1)
        decoded = Dense(units=int(input_size / factorUnits), activation='relu')(encoded)

        # decoding layers
        for i in range(hiddenLayerCount - 2, 0, -1):
            factorUnits = 2 ** i
            # print(f'{factorUnits}: {int(input_size/factorUnits)}')
            decoded = Dense(units=int(input_size / factorUnits), activation='relu')(decoded)

        # output layer
        # The output layer needs to predict the probability of an output which needs
        # to either 0 or 1 and hence we use sigmoid activation function.
        decoded = Dense(units=input_size, activation='sigmoid')(decoded)

    else:
        # output layer
        decoded = Dense(units=input_size, activation='sigmoid')(encoded)

    autoencoder = Model(inputVec, decoded)
    encoder = Model(inputVec, encoded)

    autoencoder.summary(print_fn=logging.info)
    encoder.summary(print_fn=logging.info)

    # We compile the autoencoder model with adam optimizer.
    # As fingerprint positions have a value of 0 or 1 we use binary_crossentropy as the loss function
    autoencoder.compile(optimizer=acOptimizer, loss=myloss)

    return autoencoder, encoder


def autoencoderCallback(checkpointpath: str, patience: int) -> list:
    """
    Callbacks for fitting the autoencoder
    :param checkpointpath:
    :param patience:
    :return:
    """

    # enable this checkpoint to restore the weights of the best performing model
    checkpoint = ModelCheckpoint(checkpointpath, verbose=1, period=10, save_best_only=True, mode='min',
                                 save_weights_only=True)

    # enable early stopping if val_loss is not improving anymore
    earlystop = EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)

    return [checkpoint, earlystop]


def trainfullac(df: pd.DataFrame, opts: options.TrainOptions) -> Model:
    """
    Train an autoencoder on the given feature matrix X. Response matrix is only used to
    split meaningfully in test and train data set.

    :param opts:
    :param df:
    :return: The encoder model of the trained autoencoder
    """

    # Set up the model of the AC w.r.t. the input size and the dimension of the bottle neck (z!)
    (autoencoder, encoder) = autoencoderModel(input_size=opts.fpSize, encoding_dim=opts.encFPSize)

    # if opts.acFile != "":  # don't train, use existing weights file and load it into AC model
    #     encoder.load_weights(opts.acFile)
    # else:
    # collect the callbacks for training
    callback_list = autoencoderCallback(checkpointpath=opts.outputDir + "/autoencoder.checkpointpath.hdf5", patience=20)

    # Select all fp's that are valid and turn them into a numpy array
    # This step is crucial for speed!!!
    fpMatrix = np.array(df[df["fp"].notnull()]["fp"].to_list())

    # split data into test and training data
    xTrain, xTest = train_test_split(fpMatrix, test_size=0.2, random_state=42)
    autohist = autoencoder.fit(xTrain, xTrain, callbacks=callback_list, epochs=opts.epochs, batch_size=256,
                               verbose=opts.verbose, validation_data=(xTest, xTest))
    # history
    ac_loss = autohist.history['loss']
    ac_val_loss = autohist.history['val_loss']
    ac_epochs = range(ac_loss.__len__())
    pd.DataFrame(data={'loss': ac_loss,
                       'val_loss': ac_val_loss,
                       'epoch': ac_epochs}).to_csv(opts.outputDir + "/ACmodel_trainValLoss_AC.csv", index=False)
    # generate a figure of the losses for this fold
    plt.figure()
    plt.plot(ac_epochs, ac_loss, 'bo', label='Training loss')
    plt.plot(ac_epochs, ac_val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss of AC')
    plt.legend()
    plt.savefig(fname=opts.outputDir + "/ACmodel_trainValLoss_AC.svg", format='svg')
    plt.close()
    # write the losses to .csv file for later data visualization

    # model needs to be saved and restored when predicting new input!
    # use encode() of train data as input for DL model to associate to chemical
    return encoder


def compressfingerprints(dataframe: pd.DataFrame,
                         encoder: Model) -> pd.DataFrame:
    """
    Adds a column of the compressed version of the fingerprints to the original dataframe.

    :param dataframe: Dataframe containing a column named 'fp' with the fingerprints
    :param encoder: The trained autoencoder that is used for compressing the fingerprints
    :return: The input dataframe extended by a column containing the compressed version of the fingerprints
    """

    fpMatrix = np.array(dataframe[dataframe["fp"].notnull()]["fp"].to_list())
    dataframe['fpcompressed'] = pd.Series([pd.Series(s) for s in encoder.predict(fpMatrix)])

    return dataframe
