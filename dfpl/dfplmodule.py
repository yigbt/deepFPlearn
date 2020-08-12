import argparse

# Python module for deepFPlearn tools
import re
import math
import csv
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
# %matplotlib inline
# for drawing the heatmaps
import seaborn as sns

# for fingerprint generation
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions

# for NN model functions
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras import optimizers
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold

from time import time


###############################################################################
# GENERAL FUNCTIONS --------------------------------------------------------- #

def gather(df, key, value, cols):
    """
    Simple emulation of R's gather function using pandas melt() function in python

    :param df: The data frame
    :param key: The key column name
    :param value: The value column name
    :param cols: list of column names to gather
    :return:
    """
    id_vars = [col for col in df.columns if col not in cols]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt(df, id_vars, id_values, var_name, value_name)


# ------------------------------------------------------------------------------------- #

def shuffleDataPriorToTraining(x, y):
    """
    Returns a gathered variant of the input x and y matrices
    :param x:
    :param y:
    :return:
    """

    # merge x and y
    df0 = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    # shuffle rows, drop NAs, reset index
    df1 = df0.sample(frac=1).dropna(axis=0).reset_index()

    return (df1.iloc[:, 0:x.shape[1]], df1.iloc[:, x.shape[1]:])

    # return gather(df0, key="target", value="association",
    #             cols=y.columns)


# ------------------------------------------------------------------------------------- #

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ------------------------------------------------------------------------------------- #

def smi2fp(smile, fptype, size=2048):
    """
    Convert a SMILES string to a fingerprint object of a certain type using functions
    from the RDKIT python library.

    :param smile: A single SMILES string
    :param fptype: The type of fingerprint to which the SMILES should be converted. Valid
                   values are: 'topological' (default), 'MACCS'
    :return: A fingerprint object or None, if fingerprint object could not be created
    (Respective error message is provided in STDERR).
    """
    # generate a mol object from smiles string

    # print(smile)
    cs = None
    # first transform to canoncial smiles
    try:
        cs = Chem.CanonSmiles(smile)
    except:
        print(f'[WARNING]: Not able to transform your smile to a canonical version of it: {smile}')
    if not cs:
        return None

    mol = None
    try:
        mol = Chem.MolFromSmiles(cs)
    except:
        print(
            f'[WARNING]: Not able to extract molecule from (canonically transformed) SMILES: {cs}\n          Original SMILE: {smile}')
    if not mol:
        return None

    # init fp, any better idea? e.g. calling a constructor?
    fp = Chem.Mol  # FingerprintMols.FingerprintMol(mol)

    if fptype == 'topological':  # 2048 bits
        # Topological Fingerprints:
        # The fingerprinting algorithm used is similar to that used in the Daylight
        # fingerprinter: it identifies and hashes topological paths (e.g. along bonds)
        # in the molecule and then uses them to set bits in a fingerprint of user-specified
        # lengths. After all paths have been identified, the fingerprint is typically
        # folded down until a particular density of set bits is obtained.
        try:
            # fp = Chem.RDKFingerprint(mol, fpSize=size)
            return (Chem.RDKFingerprint(mol, fpSize=size))
        except:
            print('SMILES not convertable to topological fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)
            return (None)

    elif fptype == 'MACCS':
        # MACCS Keys:
        # There is a SMARTS-based implementation of the 166 public MACCS keys.
        # The MACCS keys were critically evaluated and compared to other MACCS
        # implementations in Q3 2008. In cases where the public keys are fully defined,
        # things looked pretty good.

        try:
            # fp = MACCSkeys.GenMACCSKeys(mol)
            return (MACCSkeys.GenMACCSKeys(mol))
        except:
            print('SMILES not convertable to MACSS fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)
            return (None)

    elif fptype == 'atompairs':
        # Atom Pairs:
        # Atom-pair descriptors [3] are available in several different forms.
        # The standard form is as fingerprint including counts for each bit instead
        # of just zeros and ones. Nevertheless we use the BitVect variant here.

        try:
            # fp = Pairs.GetAtomPairFingerprintAsBitVect(mol)
            return (Pairs.GetAtomPairFingerprintAsBitVect(mol))
            # counts if features also possible here! needs to be parsed differently
            # fps.update({i:Pairs.GetAtomPairFingerprintAsIntVect(mols[i])})
        except:
            print('SMILES not convertable to atompairs fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)
            return (None)

    else:
        # Topological Torsions:
        # At the time of this writing, topological torsion fingerprints have too
        # many bits to be encodeable using the BitVector machinery, so there is no
        # GetTopologicalTorsionFingerprintAsBitVect function.

        try:
            # fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
            return (Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol))
        except:
            print('SMILES not convertable to torsions fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)
            return (None)


# ------------------------------------------------------------------------------------- #

def XandYfromInput(csvfilename: str, rtype: str, fptype: str, printfp: bool = False,
                   size: int = 2048, verbose: int = 2, returnY: bool = True) -> tuple:
    """
    Return the matrix of features for training and testing NN models (X) as numpy array.
    Provided SMILES are transformed to fingerprints, fingerprint strings are then split
    into vectors and added as row to the array which is returned.
    :param csvfilename: Filename of CSV files containing the training data. The
        SMILES/Fingerprints are stored 1st column
    :param rtype: Type of structure representation. Valid values are: 'fp' and 'smile'
    :param fptype: Type of fingerprint to be generated out
    :param printfp: Print generated fingerprints to file, namely the input file with the
        file ending '.fingerprints.csv'. Default:False
    :return: Two pandas dataframe containing the X and Y matrix for training and/or prediction. If
        no outcome data is provided, the Y matrix is a None object.
    """

    # TODOs: implement other types of fingerprint!

    df = pd.read_csv(csvfilename)
    cnames = df.columns

    if not rtype in cnames:
        print(f'[ERROR:] There is no column named {rtype} in your input training set file')
        exit(0)

    dfX = None
    if rtype == 'smiles':  # transform to canonical smiles, and then to fp
        dfX = pd.DataFrame(df['smiles'].transform(
            lambda x: np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(Chem.CanonSmiles(x)),
                                                   fpSize=size))).to_list())
    else:  # split fingerprint into matrix
        dfX = pd.DataFrame(df['fp'].transform(
            lambda x: list(x)).to_list())

    # add 'id' as rownames of dataframe
    if 'id' in cnames:
        dfX.index = df['id']

    dfY = None
    if returnY:
        # names in Y contain 'id' if present, and all other columns (=target columns)
        namesY = [c for c in cnames if c not in ['id', 'smiles', 'fp']]
        dfY = df[namesY]
        # add 'id' as rownames of dataframe
        if 'id' in cnames:
            dfY.index = df['id']

    return (dfX, dfY)


# ------------------------------------------------------------------------------------- #

def TrainingDataHeatmap(x, y):
    x['ID'] = x.index
    y['ID'] = y.index

    # xy = pd.merge(x,y,on="ID")

    # clustermap dies because of too many iterations..
    # sns.clustermap(x, metric="correlation", method="single", cmap="Blues", standard_scale=1) #, row_colors=row_colors)

    # try to cluster prior to viz
    # check this out
    # https://stackoverflow.com/questions/2455761/reordering-matrix-elements-to-reflect-column-and-row-clustering-in-naiive-python

    # viz using matplotlib heatmap https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

    return 1


# ------------------------------------------------------------------------------------- #
def removeDuplicates(x, y):
    """
    Remove duplicated feature - outcome pairs from feature matrix and outcome vector combination.

    :param x: Feature matrix
    :param y: Outcome vector

    :return: Tuple of unique Feature matrix and outcome vector combinations
    """

    # add y as additional column to x
    joined = np.append(x, [[x] for x in y], axis=1)
    # merge all columns into one string per row

    fpstrings = []
    for i in range(0, len(y)):
        fpstrings.append(["".join([str(int(c)) for c in joined[i]])])

    fpstrings_unique = np.unique(fpstrings, return_index=True)

    return (x[fpstrings_unique[1]], y[fpstrings_unique[1]])


# ------------------------------------------------------------------------------------- #

def defineCallbacks(checkpointpath: str, patience: int, rlrop: bool = False,
                    rlropfactor: float = 0.1, rlroppatience: int = 50) -> list:
    """

    :param checkpointpath:
    :param patience:
    :param rlrop:
    :param rlropfactor:
    :param rlroppatience:
    :return:
    """

    # enable this checkpoint to restore the weights of the best performing model
    checkpoint = ModelCheckpoint(checkpointpath, monitor='val_loss', verbose=1, period=10,
                                 save_best_only=True, mode='min', save_weights_only=True)

    # enable early stopping if val_loss is not improving anymore
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=patience,
                              verbose=1,
                              restore_best_weights=True)

    callbacks = []
    if rlrop:
        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=rlropfactor, patience=rlroppatience)
        callbacks = [checkpoint, earlystop, rlrop]
    else:
        callbacks = [checkpoint, earlystop]

    # Return list of callbacks - collect the callbacks for training
    return (callbacks)


# ------------------------------------------------------------------------------------- #

def defineNNmodelMulti(inputSize=2048, outputSize=None, l2reg=0.001, dropout=0.2,
                       activation='relu', optimizer='Adam', lr=0.001, decay=0.01):
    if optimizer == 'Adam':
        myoptimizer = optimizers.Adam(learning_rate=lr, decay=decay)  # , beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer == 'SGD':
        myoptimizer = SGD(lr=lr, momentum=0.9, decay=decay)
    else:
        myoptimizer = optimizer

    nhl = int(math.log2(inputSize) / 2 - 1)

    model = Sequential()
    # From input to 1st hidden layer
    model.add(Dense(units=int(inputSize / 2), input_dim=inputSize,
                    activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    # next hidden layers
    for i in range(1, nhl):
        factorunits = 2 ** (i + 1)
        factordropout = 2 * i
        model.add(Dense(units=int(inputSize / factorunits),
                        activation=activation,
                        kernel_regularizer=regularizers.l2(l2reg)))
        model.add(Dropout(dropout / factordropout))
    # multi-class output layer
    # use sigmoid to get independent probabilities for each output node
    # (need not add up to one, as they would using softmax)
    # https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
    model.add(Dense(units=outputSize, activation='sigmoid'))

    model.summary()

    # compile model
    model.compile(loss="binary_crossentropy", optimizer=myoptimizer, metrics=['accuracy'])

    return model


# ------------------------------------------------------------------------------------- #

def defineNNmodel(inputSize: int = 2048, l2reg: float = 0.001, dropout: float = 0.2,
                  activation: str = 'relu', optimizer: str = 'Adam', lr: float = 0.001,
                  decay: float = 0.01) -> Model:
    """

    :param inputSize:
    :param l2reg:
    :param dropout:
    :param activation:
    :param optimizer:
    :param lr:
    :param decay:
    :return:
    """

    if optimizer == 'Adam':
        myoptimizer = optimizers.Adam(learning_rate=lr, decay=decay)  # , beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer == 'SGD':
        myoptimizer = SGD(lr=lr, momentum=0.9, decay=decay)
    else:
        myoptimizer = optimizer

    myhiddenlayers = {"2048": 6, "1024": 5, "999": 5, "512": 4, "256": 3}

    if not str(inputSize) in myhiddenlayers.keys():
        print("Wrong inputsize. Must be in {2048, 1024, 999, 512, 256}.")
        return None

    nhl = int(math.log2(inputSize) / 2 - 1)

    model = Sequential()
    # From input to 1st hidden layer
    model.add(Dense(units=int(inputSize / 2), input_dim=inputSize,
                    activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    # next hidden layers
    for i in range(1, nhl):
        factorunits = 2 ** (i + 1)
        factordropout = 2 * i
        model.add(Dense(units=int(inputSize / factorunits),
                        activation=activation,
                        kernel_regularizer=regularizers.l2(l2reg)))
        model.add(Dropout(dropout / factordropout))
    # output layer
    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()

    # compile model
    model.compile(loss="mse", optimizer=myoptimizer, metrics=['accuracy'])

    return model


# ------------------------------------------------------------------------------------- #

def autoencoderModel(input_size: int = 2048, encoding_dim: int = 256, myloss: object = 'binary_crossentropy',
                     mylr: float = 0.001, mydecay: float = 0.01) -> tuple:
    """
    This function provides an autoencoder model to reduce a certain input to a compressed version.

    :param encoding_dim: Size of the compressed representation. Default: 85
    :param input_size: Size of the input. Default: 2048
    :param myactivation: Activation function, see Keras activation functions for potential values. Default: relu
    :param myoptimizer: Optimizer, see Keras optmizers for potential values. Default: adadelta
    :param myloss: Loss function, see Keras Loss functions for potential values. Default: binary_crossentropy
    :return: a tuple of autoencoder and encoder models
    """

    myoptimizer = optimizers.Adam(learning_rate=mylr, decay=mydecay)  # , beta_1=0.9, beta_2=0.999, amsgrad=False)

    # get the number of meaningful hidden layers (latent space included)
    nhl = round(math.log2(input_size / encoding_dim))

    # the input placeholder
    input_vec = Input(shape=(input_size,))

    # 1st hidden layer, that receives weights from input layer
    # equals bottle neck layer, if nhl==1!
    encoded = Dense(units=int(input_size / 2), activation='relu')(input_vec)

    if nhl > 1:
        # encoding layers, incl. bottle neck
        for i in range(1, nhl):
            factorunits = 2 ** (i + 1)
            # print(f'{factorunits}: {int(input_size / factorunits)}')
            encoded = Dense(units=int(input_size / factorunits), activation='relu')(encoded)

        #        encoding_dim = int(input_size/factorunits)

        # 1st decoding layer
        factorunits = 2 ** (nhl - 1)
        decoded = Dense(units=int(input_size / factorunits), activation='relu')(encoded)

        # decoding layers
        for i in range(nhl - 2, 0, -1):
            factorunits = 2 ** i
            # print(f'{factorunits}: {int(input_size/factorunits)}')
            decoded = Dense(units=int(input_size / factorunits), activation='relu')(decoded)

        # output layer
        # The output layer needs to predict the probability of an output which needs to either 0 or 1 and hence we use sigmoid activation function.
        decoded = Dense(units=input_size, activation='sigmoid')(decoded)

    else:
        # output layer
        decoded = Dense(units=input_size, activation='sigmoid')(encoded)

    autoencoder = Model(input_vec, decoded)
    encoder = Model(input_vec, encoded)

    autoencoder.summary()
    encoder.summary()

    # We compile the autoencoder model with adam optimizer.
    # As fingerprint positions have a value of 0 or 1 we use binary_crossentropy as the loss function
    autoencoder.compile(optimizer=myoptimizer, loss=myloss)

    return (autoencoder, encoder)


# ------------------------------------------------------------------------------------- #

# def predict_values(ac_model_file_path, model_file_path, pdx):
#     """
#     Predict a set of chemicals using a selected model.
#
#     :param model_file_path: Path to the model weights of the prediction DNN
#     :param pdx: A matrix containing the fingerprints of the chemicals, generated via XfromInput function
#     :return: A dataframe of 2 columns: random - predicted values using random model, trained - predicted values
#     using trained model. Rownames are consecutive numbers of the input rows, or if provided in the input file
#     the values of the id column
#     """
#
#     # ac_model_file_path="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/model.2048.256.ER.checkpoint.AC-model.hdf5"
#     # model_file_path  ="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/model.2048.256.ER.checkpoint.model.hdf5"
#
#     print(f"[INFO:] Loaded model weights for autoencoder: '{model_file_path}'")
#     print(f"[INFO:] Loaded model weights for prediction DNN: '{model_file_path}'")
#
#     start = time()
#     # create autoencoder
#     (autoencoder, encoder) = autoencoderModel(input_size=pdx.shape[1], encoding_dim=256)
#     # load AC weights
#     autoencoder.load_weights(ac_model_file_path)
#     # encode input
#     encodedX = encoder.predict(pdx)
#
#     trainTime = str(round((time() - start) / 60, ndigits=2))
#     print(f"[INFO:] Computation time used for encoding: {trainTime} min")
#
#     start = time()
#     # create DNN
#     predictor = defineNNmodel(inputSize=encodedX.shape[1])
#     # predict with random weights
#     predictions_random = predictor.predict(encodedX)
#     # load DNN weights
#     predictor.load_weights(model_file_path)
#     # predict encoded input
#     predictions = predictor.predict(encodedX)
#
#     trainTime = str(round((time() - start) / 60, ndigits=2))
#     print(f"[INFO:] Computation time used for the predictions: {trainTime} min")
#
#     df = pd.DataFrame(data={'random': predictions_random.flatten(),
#                             'trained': predictions.flatten()},
#                       columns=['random', 'trained'],
#                       index=pdx.index)
#     print(df)
#     return (df)


# ------------------------------------------------------------------------------------- #

def trainfullac(X: pd.DataFrame, y: pd.DataFrame, useweights: str = None, epochs: int = 0,
                encdim: int = 256, checkpointpath: str = None, verbose: int = 0) -> Model:
    """
    Train an autoencoder on the given feature matrix X. Response matrix is only used to
    split meaningfully in test and train data set.

    :param X: Feature matrix.
    :param y: Matrix containing the response variables.
    :param epochs: Number of epochs to train the autoencoder
    :param encdim: Dimension of latent space. This and the column dimension of X control
                   the number and size of the hidden layers.
    :param checkpointpath: path to prefix the model files during callbacks
    :param verbose: Verbosity level. Is forwarded to the Keras fit function that prints
                    training states in different verbosity levels
    :return: The encoder model of the trained autoencoder
    """

    # Set up the model of the AC w.r.t. the input size and the dimension of the bottle neck (z!)
    (autoencoder, encoder) = autoencoderModel(input_size=X.shape[1], encoding_dim=encdim)

    if useweights:  # don't train, use existing weights file and load it into AC model
        #autoencoder.load_weights(useweights)
        encoder.load_weights(useweights)
    else:
        # collect the callbacks for training
        callback_list = defineCallbacks(checkpointpath=checkpointpath,
                                        patience=20, rlrop=False)

        # split data into test and training data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Fit the AC
        autohist = autoencoder.fit(X_train, X_train,
                                   callbacks=callback_list,
                                   epochs=epochs,
                                   batch_size=256,
                                   shuffle=True,
                                   verbose=verbose,
                                   validation_data=(X_test, X_test))
        # history
        ac_loss = autohist.history['loss']
        ac_val_loss = autohist.history['val_loss']
        ac_epochs = range(ac_loss.__len__())
        pd.DataFrame(data={'loss': ac_loss,
                           'val_loss': ac_val_loss,
                           'epoch': ac_epochs}).to_csv(checkpointpath.replace(".hdf5",
                                                                              "_trainValLoss_AC.csv"), index=False)
        # generate a figure of the losses for this fold
        plt.figure()
        plt.plot(ac_epochs, ac_loss, 'bo', label='Training loss')
        plt.plot(ac_epochs, ac_val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss of AC')
        plt.legend()
        plt.savefig(fname=checkpointpath.replace(".hdf5",
                                                 "_trainValLoss_AC.svg"),
                    format='svg')
        plt.close()
        # write the losses to .csv file for later data visualization

    # model needs to be saved and restored when predicting new input!
    # use encode() of train data as input for DL model to associate to chemical
    return encoder

# ------------------------------------------------------------------------------------- #

def plotTrainHistory(hist, target, fileAccuracy, fileLoss):
    """
    Plot the training performance in terms of accuracy and loss values for each epoch.
    :param hist: The history returned by model.fit function
    :param target: The name of the target of the model
    :param fileAccuracy: The filename for plotting accuracy values
    :param fileLoss: The filename for plotting loss values
    :return: none
    """

    # plot accuracy
    plt.figure()
    plt.plot(hist.history['accuracy'])
    if 'val_accuracy' in hist.history.keys():
        plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy - ' + target)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if 'val_accuracy' in hist.history.keys():
        plt.legend(['Train', 'Test'], loc='upper left')
    else:
        plt.legend(['Train'], loc='upper_left')
    plt.savefig(fname=fileAccuracy, format='svg')

    # Plot training & validation loss values
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss - ' + target)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    #        plt.show()
    plt.savefig(fname=fileLoss, format='svg')
    plt.close()


# ------------------------------------------------------------------------------------- #

def plotAUC(fpr, tpr, auc, target, filename, title=""):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve {target}')
    plt.legend(loc='best')
    plt.savefig(fname=filename, format='svg')
    plt.close()


# ------------------------------------------------------------------------------------- #

def plotHeatmap(matrix, filename, title=""):
    plt.figure()
    plt.imshow(matrix, cmap='Greys', interpolation='nearest')
    plt.title(title)
    plt.savefig(fname=filename, format='svg')
    plt.close()


# ------------------------------------------------------------------------------------- #

# plot model history more easily, see whyboris@github: https://gist.github.com/whyboris/91ee793ddc92cf1e824978cf31bb790c
# when plotting, smooth out the points by some factor (0.5 = rough, 0.99 = smooth)
# method taken from `Deep Learning with Python` by François Chollet

# 01 --------------------------------------------------------------------------------- #

# def smooth_curve(points, factor=0.75):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points
#
#
# # 02 ---------------------------------------------------------------------------------- #
#
# def set_plot_history_data(ax, history, which_graph):
#     if which_graph == 'acc':
#         train = smooth_curve(history.history['accuracy'])
#         valid = smooth_curve(history.history['val_accuracy'])
#
#     if which_graph == 'loss':
#         train = smooth_curve(history.history['loss'])
#         valid = smooth_curve(history.history['val_loss'])
#
#     # plt.xkcd() # make plots look like xkcd
#
#     epochs = range(1, len(train) + 1)
#
#     trim = 0  # remove first 5 epochs
#     # when graphing loss the first few epochs may skew the (loss) graph
#
#     ax.plot(epochs[trim:], train[trim:], 'dodgerblue', linewidth=15, alpha=0.1)
#     ax.plot(epochs[trim:], train[trim:], 'dodgerblue', label=('Training'))
#
#     ax.plot(epochs[trim:], valid[trim:], 'g', linewidth=15, alpha=0.1)
#     ax.plot(epochs[trim:], valid[trim:], 'g', label=('Validation'))
#

# 03 ---------------------------------------------------------------------------------- #
#
# def get_max_validation_accuracy(history):
#     validation = smooth_curve(history.history['val_accuracy'])
#     ymax = max(validation)
#     return 'Max validation accuracy ≈ ' + str(round(ymax, 3) * 100) + '%'
#
#
# def get_max_training_accuracy(history):
#     training = smooth_curve(history.history['accuracy'])
#     ymax = max(training)
#     return 'Max training accuracy ≈ ' + str(round(ymax, 3) * 100) + '%'
#

# 04---------------------------------------------------------------------------------- #

# def plot_history(history, file):
#     fig, (ax1, ax2) = plt.subplots(nrows=2,
#                                    ncols=1,
#                                    figsize=(10, 6),
#                                    sharex=True,
#                                    gridspec_kw={'height_ratios': [5, 2]})
#
#     set_plot_history_data(ax1, history, 'acc')
#
#     set_plot_history_data(ax2, history, 'loss')
#
#     # Accuracy graph
#     ax1.set_ylabel('Accuracy')
#     ax1.set_ylim(bottom=0.5, top=1)
#     ax1.legend(loc="lower right")
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.xaxis.set_ticks_position('none')
#     ax1.spines['bottom'].set_visible(False)
#
#     # max accuracy text
#     plt.text(0.5,
#              0.6,
#              get_max_validation_accuracy(history),
#              horizontalalignment='right',
#              verticalalignment='top',
#              transform=ax1.transAxes,
#              fontsize=12)
#     plt.text(0.5,
#              0.8,
#              get_max_training_accuracy(history),
#              horizontalalignment='right',
#              verticalalignment='top',
#              transform=ax1.transAxes,
#              fontsize=12)
#
#     # Loss graph
#     ax2.set_ylabel('Loss')
#     ax2.set_yticks([])
#     ax2.plot(legend=False)
#     ax2.set_xlabel('Epochs')
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
#
#     plt.tight_layout()
#     plt.savefig(fname=file, format='svg')
#     plt.close()
#

# ------------------------------------------------------------------------------------- #

def drawHeatmap(data, anno):
    # (data=pd.DataFrame(Xt), anno = pd.DataFrame(Yt.astype(int)))

    # annotation bar colors
    my_ann_colors = dict(zip(anno[0].unique(), ["blue", "red"]))
    row_colors = anno[0].map(my_ann_colors)

    cl = sns.clustermap(data, metric="euclidean", method="single", z_score=None, standard_scale=None,
                        col_cluster=False, cmap="Greys", row_colors=row_colors, yticklabels=False)
    cl.fig.suptitle('Distributions of [1,0] in fingerprints of target: AR')

    url = 'https://python-graph-gallery.com/wp-content/uploads/mtcars.csv'
    df = pd.read_csv(url)
    # set df index using existing columns
    df = df.set_index('model')
    # remove name of index column, numbered rows have been the index before, they are gone now
    del df.index.name
    df
    my_palette = dict(zip(df.cyl.unique(), ["orange", "yellow", "brown"]))
    row_colors = df.cyl.map(my_palette)

    # Default plot
    # sns.clustermap(df)

    # Clustermethods
    my_palette = dict()
    sns.clustermap(df, metric="correlation", standard_scale=1, method="single", cmap="Blues", row_colors=row_colors)


###############################################################################
# TRAIN FUNCTIONS --------------------------------------------------------- #


def defineOutfileNames(pathprefix: str, target: str, fold: int) -> tuple:
    """
    This function returns the required paths for output files or directories.

    :param pathprefix: A file path prefix for all files.
    :param mtype: The model type. Its set by the trainNNmodels function with information on autoencoder or not,
    and if AC is used, then with its parameters.
    :param target: The name of the target.

    :return: A tuple of 14 output file names.
    """

    modelname = target + '.Fold-' + str(fold)

    modelfilepathW = str(pathprefix) + '.' + modelname + '.weights.h5'
    modelfilepathM = str(pathprefix) + '.' + modelname + '.json'
    modelhistplotpathL = str(pathprefix) + '.' + modelname + '.loss.svg'
    modelhistplotpathA = str(pathprefix) + '.' + modelname + '.acc.svg'
    modelhistplotpath = str(pathprefix) + '.' + modelname + '.history.svg'
    modelhistcsvpath = str(pathprefix) + '.' + modelname + '.history.csv'
    modelvalidation = str(pathprefix) + '.' + modelname + '.validation.csv'
    modelAUCfile = str(pathprefix) + '.' + modelname + '.auc_value.svg'
    modelAUCfiledata = str(pathprefix) + '.' + modelname + '.auc_value.data.csv'
    outfilepath = str(pathprefix) + '.' + modelname + '.trainingResults.txt'
    checkpointpath = str(pathprefix) + '.' + modelname + '.checkpoint.model.hdf5'
    modelheatmapX = str(pathprefix) + '.' + modelname + '.heatmap.X.svg'
    modelheatmapZ = str(pathprefix) + '.' + modelname + '.AC.heatmap.Z.svg'

    return (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
            modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
            modelAUCfiledata, outfilepath, checkpointpath,
            modelheatmapX, modelheatmapZ)


# ------------------------------------------------------------------------------------- #

def eval01Distributions(Xt, Yt, y_train, y_test, verbosity=0):
    """
    Evaluate the percentage of 0 values in the outcome variable of the whole dataset and the splitted (train,test)
    dataset, and the percentage of 0 values in the feature matrix.

    :param Xt: The whole feature matrix
    :param Yt: The whole outcome vector
    :param y_train: The outcome vector of the training set
    :param y_test: The outcome vector of the test set
    :param verbosity: The verbosity level. Info is only printed if verbosity is not 0.

    :return: Nothing is returned.
    """

    if verbosity == 0:
        return
    else:
        unique, counts = np.unique(Yt, return_counts=True)
        perc = round(100 / len(Yt) * counts[1])
        print(f"[INFO:] Percentage of '1' values in outcome variable (whole dataset): {perc}\n")

        uniqueRtr, countsRtr = np.unique(y_train, return_counts=True)
        uniqueRte, countsRte = np.unique(y_test, return_counts=True)
        perc = round(100 / len(y_train) * countsRtr[1])
        print(f"[INFO:] Percentage of '1' values in training outcomes: {perc}\n")
        perc = round(100 / len(y_test) * countsRte[1])
        print(f"[INFO:] Percentage of '1' values in test outcomes: {perc}\n")

        print(
            f"[INFO:] Average percentage of '0' positions in fingerprints: {round(np.sum(Xt == 0) / (np.sum(Xt == 0) + np.sum(Xt == 1)), ndigits=4)}")
    return


# ------------------------------------------------------------------------------------- #

def prepareDataSet(y: pd.DataFrame, x: pd.DataFrame, t: str) -> tuple:
    """
    A function to remove NA values from the output column, duplicates from input and output
    and to transform the data into numpy arrays for keras functions.

    :param y: The target vector that may contain NAs.
    :param x: The input matrix that may contain duplicates (including resp. target value!)
    :param t: The target string
    :return: Tuple of x and y ready for splitting into test and train sets for ML with keras.
    """

    # which rows contain 'NA' in target column
    tmp = y[t].astype('category')
    Y = np.asarray(tmp)
    naRows = np.isnan(Y)

    # transform pd dataframe to numpy array for keras
    X = x.to_numpy()

    # subset data according to target non-NA values
    Ytall = Y[~naRows]
    Xtall = X[~naRows]

    # remove all duplicated feature values - outcome pairs, and
    # shuffle data, such that validation_split in fit function selects a random subset!
    # Remember, that it takes the fraction for validation from the end of our data!

    (XtU, YtU) = removeDuplicates(x=Xtall, y=Ytall)

    # return(shuffleDataPriorToTraining(x=XtU, y=YtU))
    return (XtU, YtU)


# ------------------------------------------------------------------------------------- #

# def plot_history_vis(hist, model_hist_plot_path, model_hist_csv_path,
#                    model_hist_plot_path_a, model_hist_plot_path_l, target):
#     plot_history(history=hist, file=model_hist_plot_path)
#     histDF = pd.DataFrame(hist.history)
#     histDF.to_csv(model_hist_csv_path)
#
#     # plot accuracy and loss for the training and validation during training
#     plotTrainHistory(hist=hist, target=target,
#                      file_accuracy=model_hist_plot_path_a,
#                      file_loss=model_hist_plot_path_l)
#

# ------------------------------------------------------------------------------------- #
#
# def validate_multi_model_on_test_data(x_test, checkpoint_path, y_test, col_names, result_file):
#     # load checkpoint model with min(val_loss)
#     trainedmodel = define_nn_model_multi(input_size=x_test.shape[1], output_size=y_test.shape[1])
#
#     # predict values with random model
#     predictions_random = pd.DataFrame(trainedmodel.predict(x_test), columns=col_names + '-predRandom')
#
#     # load weights into random model
#     trainedmodel.load_weights(checkpoint_path)
#
#     # predict with trained model
#     predictions = pd.DataFrame(trainedmodel.predict(x_test),
#                                columns=col_names + '-pred')
#     scores = pd.DataFrame((predictions.round() == y_test).sum() / y_test.shape[0], columns=['correctPredictions'])
#
#     results = pd.concat([predictions_random, predictions, pd.DataFrame(y_test, columns=col_names + '-true')], axis=1)
#     results.to_csv(result_file)
#
#     return scores


# ------------------------------------------------------------------------------------- #

# def validate_model_on_test_data(x_test, checkpoint_path, y_test, model_type, model_validation, target,
#                             model_auc_file_data, model_auc_file):
#     """
#     Function that validates trained model with test data set. History and AUC plots are generated.
#     Accuracy and Loss of model on test data, as well as MCC and confusion matrix is calculated and returned.
#
#     :param x_test:
#     :param checkpoint_path:
#     :param y_test:
#     :param model_type:
#     :param model_validation:
#     :param target:
#     :param model_auc_file_data:
#     :param model_auc_file:
#     :return: Tupel containing Loss, Accuracy, MCC, tn, fp, fn, tp of trained model on test data.
#     """
#
#     # load checkpoint model with min(val_loss)
#     trainedmodel = define_nn_model(input_size=x_test.shape[1])
#
#     # predict values with random model
#     predictions_random = pd.DataFrame(trainedmodel.predict(x_test))
#
#     # load weights into random model
#     trainedmodel.load_weights(checkpoint_path)
#     # predict with trained model
#     predictions = pd.DataFrame(trainedmodel.predict(x_test))
#
#     # save validation data to .csv file
#     validation = pd.DataFrame({'predicted': predictions[0].ravel(),
#                                'true': list(y_test),
#                                'predicted_random': predictions_random[0].ravel(),
#                                'model_type': model_type})
#     validation.to_csv(model_validation)
#
#     # compute MCC
#     predictionsInt = [int(round(x)) for x in predictions[0].ravel()]
#     ytrueInt = [int(y) for y in y_test]
#     MCC = matthews_corrcoef(ytrueInt, predictionsInt)
#
#     # generate the AUC-ROC curve data from the validation data
#     fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, predictions, drop_intermediate=False)
#
#     auc_keras = auc_value(fpr_keras, tpr_keras)
#
#     aucdata = pd.DataFrame(list(zip(fpr_keras,
#                                     tpr_keras,
#                                     [auc_keras for x in range(1, len(fpr_keras))],
#                                     [target for x in range(1, len(fpr_keras))])),
#                            columns=['fpr', 'tpr', 'auc_value', 'target'])
#     aucdata.to_csv(model_auc_file_data)
#
#     plotAUC(fpr=fpr_keras, tpr=tpr_keras, target=target, auc_value=auc_keras, filename=model_auc_file)
#
#     # [[tn, fp]
#     #  [fn, tp]]
#     cfm = confusion_matrix(y_true=ytrueInt, y_pred=predictionsInt)
#
#     scores = trainedmodel.evaluate(x_test, y_test, verbose=0)
#
#     print(f'TARGET: {target} Loss: {scores[0].__round__(2)} Acc: {scores[1].__round__(2)}')
#     print(f'MCC: {MCC.__round__(2)}')
#     print(f'CFM: \n\tTN={cfm[0][0]}\tFP={cfm[0][1]}\n\tFN={cfm[1][0]}\tTP={cfm[1][1]}')
#
#     return (scores[0], scores[1], MCC, cfm[0][0], cfm[0][1], cfm[1][0], cfm[1][1])


# ------------------------------------------------------------------------------------- #
# def train_nn_models_multi(modelfilepathprefix: str, x: pd.DataFrame, y: pd.DataFrame,
#                        split: float = 0.2, epochs: int = 500,
#                        verbose: int= 2, kfold: int = 5) -> None:
#     # remove 'id' column if present
#     if 'id' in x.columns:
#         x = x.drop('id', axis=1)
#     if 'id' in y.columns:
#         y = y.drop('id', axis=1)
#
#     # drop compounds that are not measured for all target columns, transform to numpy
#     (xmulti, ymulti) = shuffleDataPriorToTraining(x, y)
#
#     # do a kfold cross validation for the autoencoder training
#     kfoldCValidator = KFold(n_splits=kfold, shuffle=True, random_state=42)
#
#     # store acc and loss for each fold
#     allscores = pd.DataFrame(columns=["fold_no",  # fold number of k-fold CV
#                                       "loss", "val_loss", "acc", "val_acc",  # FNN training
#                                       "loss_test", "acc_test", "mcc_test"])  # FNN test data
#
#     fold_no = 1
#
#     # split the data
#     for train, test in kfoldCValidator.split(xmulti, ymulti):
#
#         # define all the output file/path names
#         (modelfilepathW, modelfilepathM, model_hist_plot_path_l, model_hist_plot_path_a,
#          model_hist_plot_path, model_hist_csv_path, model_validation, model_auc_file,
#          model_auc_file_data, outfilepath, checkpoint_path,
#          modelheatmapX, modelheatmapZ) = define_out_file_names(path_prefix=modelfilepathprefix,
#                                                             target="multi", fold=fold_no)
#
#         # use a dnn for multi-class prediction
#         model = define_nn_model_multi(input_size=xmulti[train].shape[1], output_size=ymulti.shape[1])
#
#         callback_list = defineCallbacks(checkpoint_path=checkpoint_path, patience=20,
#                                         rlrop=True, rlropfactor=0.1, rlroppatience=100)
#         # measure the training time
#         start = time()
#
#         # train and validate
#         hist = model.fit(xmulti[train], ymulti[train],
#                          callbacks=callback_list,
#                          epochs=epochs, batch_size=256, verbose=2, validation_split=split)
#
#         trainTime = str(round((time() - start) / 60, ndigits=2))
#
#         if verbose > 0:
#             print(f"[INFO:] Computation time for training the multi-label FNN: {trainTime} min")
#
#         # validate model on test data set (x_test, y_test)
#         scores = validate_multi_model_on_test_data(x_test=xmulti[test],
#                                               checkpoint_path=checkpoint_path,
#                                               y_test=ymulti[test],
#                                               col_names=y.columns,
#                                               result_file=outfilepath.replace("trainingResults.txt",
#                                                                              "predictionResults.csv"))
#
#         idx = hist.history['val_loss'].index(min(hist.history['val_loss']))
#         row_df = pd.DataFrame([[fold_no,
#                                 hist.history['loss'][idx], hist.history['val_loss'][idx],
#                                 hist.history['accuracy'][idx], hist.history['val_accuracy'][idx],
#                                 scores[0], scores[1], scores[2]]],
#                               columns=["fold_no",  # fold number of k-fold CV
#                                        "loss", "val_loss", "acc", "val_acc",  # FNN training
#                                        "loss_test", "acc_test", "mcc_test"]
#                               )
#         print(row_df)
#         allscores = allscores.append(row_df, ignore_index=True)
#
#         fold_no = fold_no + 1
#         del model
#
#     print(allscores)
#
#     # finalize model
#     # 1. provide best performing fold variant
#     # select best model based on MCC
#     idx2 = allscores[['mcc_test']].idxmax().ravel()[0]
#     fold_no = allscores._get_value(idx2, 'fold_no')
#
#     modelname = 'multi.Fold-' + str(fold_no)
#     checkpoint_path = str(modelfilepathprefix) + '.' + modelname + '.checkpoint.model.hdf5'
#     bestModelfile = checkpoint_path.replace("Fold-" + str(fold_no) + ".checkpoint.", "best.FNN-")
#
#     file = re.sub("\.hdf5", "scores.csv", re.sub("Fold-.\.checkpoint", "Fold-All", checkpoint_path))
#     allscores.to_csv(file)
#
#     # copy best DNN model
#     shutil.copyfile(checkpoint_path, bestModelfile)
#     print(f'[INFO]: Best models for FNN is saved:\n        - {bestModelfile}')
#
#     # AND retrain with full data set
#     fullModelfile = checkpoint_path.replace("Fold-" + str(fold_no) + ".checkpoint", "full.FNN-")
#     # measure the training time
#     start = time()
#
#     model = define_nn_model(input_size=xmulti[train].shape[1])
#     callback_list = defineCallbacks(checkpoint_path=fullModelfile, patience=20,
#                                     rlrop=True, rlropfactor=0.1, rlroppatience=100)
#     # train and validate
#     hist = model.fit(xmulti, ymulti,
#                      callbacks=callback_list,
#                      epochs=epochs, batch_size=256, verbose=2, validation_split=split)
#     #                             validation_data=(x_test, y_test))  # this overwrites val_split!
#     trainTime = str(round((time() - start) / 60, ndigits=2))
#
#     if verbose > 0:
#         print(f"[INFO:] Computation time for training the full classification FNN: {trainTime} min")
#     plot_history_vis(hist,
#                    model_hist_plot_path.replace("Fold-" + str(fold_no), "full.DNN-model"),
#                    model_hist_csv_path.replace("Fold-" + str(fold_no), "full.DNN-model"),
#                    model_hist_plot_path_a.replace("Fold-" + str(fold_no), "full.DNN-model"),
#                    model_hist_plot_path_l.replace("Fold-" + str(fold_no), "full.DNN-model"), target)
#     print(f'[INFO]: Full models for DNN is saved:\n        - {fullModelfile}')
#
#     pd.DataFrame(hist.history).to_csv(fullModelfile.replace(".hdf5", ".history.csv"))
#     # stats.append([target, [x.__round__(2) for x in scores]])


# ------------------------------------------------------------------------------------- #

# def trainNNmodels(modelfilepathprefix: str, x: pd.DataFrame, y: pd.DataFrame,
#                   split: float = 0.2, epochs: int = 50, params: str = None,
#                   verbose: int = 2, kfold: int = 5) -> None:
#     """
#     Train individual models for all targets (columns) present in the provided target data (y) and a multi-label
#     model that classifies all targets at once. For each individual target the data is first subsetted to exclude NA
#     values (for target associations). A random sample of the remaining data (size is the split fraction) is used for
#     training and the remaining data for validation.
#
#     :param modelfilepathprefix: A path prefix for all output files
#     :param x: The feature matrix.
#     :param y: The outcome matrix.
#     :param split: The percentage of data used for validation.
#     :param epochs: The number of epochs for training the autoencoder and the DNN for classification.
#     Note: Early stopping and fallback is enabled.
#     :param params: A .csv files containing paramters that should be evaluated. See file tunedParams.csv.
#     :param verbose: Verbosity level.
#
#     :return: A list with loss and accuracy values for each individual model.
#     """
#
#     # remove 'id' column if present
#     if 'id' in x.columns:
#         x = x.drop('id', axis=1)
#     if 'id' in y.columns:
#         y = y.drop('id', axis=1)
#
#     if params:
#         parameters = pd.read_csv(params)
#
#     # add a target 'summarized' for association to ANY of the target genes
#     # maybe it improves the detection of '1's due to increased data set
#     mysum = y.sum(axis=1)
#     y['summarized'] = [0 if s == 0 else 1 for s in mysum]
#
#     ### For each individual target (+ summarized target)
#     for target in y.columns:  # [:1]:
#         # target=y.columns[0] # --> only for testing the code
#
#         # rm NAs and duplicates, shuffle, and transform to numpy arrays
#         (Xt, Yt) = prepareDataSet(y, x, target)
#
#         # do a kfold cross validation for the FNN training
#         kfoldCValidator = KFold(n_splits=kfold, shuffle=True, random_state=42)
#
#         # store acc and loss for each fold
#         allscores = pd.DataFrame(columns=["fold_no",  # fold number of k-fold CV
#                                           "loss", "val_loss", "acc", "val_acc",  # FNN training
#                                           "loss_test", "acc_test", "mcc_test"])  # FNN test data
#
#         fold_no = 1
#
#         # split the data
#         for train, test in kfoldCValidator.split(Xt, Yt):
#
#             if verbose > 0:
#                 print(f'[INFO]: Training of fold number: {fold_no} ------------------------------------\n')
#
#             # define all the output file/path names
#             (modelfilepathW, modelfilepathM, model_hist_plot_path_l, model_hist_plot_path_a,
#              model_hist_plot_path, model_hist_csv_path, model_validation, model_auc_file,
#              model_auc_file_data, outfilepath, checkpoint_path,
#              modelheatmapX, modelheatmapZ) = define_out_file_names(path_prefix=modelfilepathprefix,
#                                                                 target=target, fold=fold_no)
#
#             model = define_nn_model(input_size=Xt[train].shape[1])
#
#             callback_list = defineCallbacks(checkpoint_path=checkpoint_path, patience=20,
#                                             rlrop=True, rlropfactor=0.1, rlroppatience=100)
#             # measure the training time
#             start = time()
#             # train and validate
#             hist = model.fit(Xt[train], Yt[train],
#                              callbacks=callback_list,
#                              epochs=epochs, batch_size=256, verbose=2, validation_split=split)
#             #                             validation_data=(x_test, y_test))  # this overwrites val_split!
#             trainTime = str(round((time() - start) / 60, ndigits=2))
#
#             if verbose > 0:
#                 print(f"[INFO:] Computation time for training the single-label FNN: {trainTime} min")
#
#             # validate model on test data set (x_test, y_test)
#             scores = validate_model_on_test_data(Xt[test], checkpoint_path, Yt[test],
#                                              "FNN", model_validation, target,
#                                              model_auc_file_data, model_auc_file)
#
#             idx = hist.history['val_loss'].index(min(hist.history['val_loss']))
#
#             row_df = pd.DataFrame([[fold_no,
#                                     hist.history['loss'][idx], hist.history['val_loss'][idx],
#                                     hist.history['accuracy'][idx], hist.history['val_accuracy'][idx],
#                                     scores[0], scores[1], scores[2]]],
#                                   columns=["fold_no",  # fold number of k-fold CV
#                                            "loss", "val_loss", "acc", "val_acc",  # FNN training
#                                            "loss_test", "acc_test", "mcc_test"]
#                                   )
#             print(row_df)
#             allscores = allscores.append(row_df, ignore_index=True)
#             fold_no = fold_no + 1
#             del model
#             # now next fold
#
#         print(allscores)
#
#         # finalize model
#         # 1. provide best performing fold variant
#         # select best model based on MCC
#         idx2 = allscores[['mcc_test']].idxmax().ravel()[0]
#         fold_no = allscores._get_value(idx2, 'fold_no')
#
#         modelname = target + '.Fold-' + str(fold_no)
#         checkpoint_path = str(modelfilepathprefix) + '.' + modelname + '.checkpoint.model.hdf5'
#         bestModelfile = checkpoint_path.replace("Fold-" + str(fold_no) + ".checkpoint.", "best.FNN-")
#
#         # store all scores
#         file = re.sub("\.hdf5", "scores.csv", re.sub("Fold-.\.checkpoint", "Fold-All", checkpoint_path))
#         allscores.to_csv(file)
#
#         # copy best DNN model
#         shutil.copyfile(checkpoint_path, bestModelfile)
#         print(f'[INFO]: Best model for FNN is saved:\n        - {bestModelfile}')
#
#         # AND retrain with full data set
#         fullModelfile = checkpoint_path.replace("Fold-" + str(fold_no) + ".checkpoint", "full.FNN-")
#         # measure the training time
#         start = time()
#
#         model = define_nn_model(input_size=Xt.shape[1])  # X_train.shape[1])
#         callback_list = defineCallbacks(checkpoint_path=fullModelfile, patience=20,
#                                         rlrop=True, rlropfactor=0.1, rlroppatience=100)
#         # train and validate
#         hist = model.fit(Xt, Yt,
#                          callbacks=callback_list,
#                          epochs=epochs, batch_size=256, verbose=2, validation_split=split)
#         #                             validation_data=(x_test, y_test))  # this overwrites val_split!
#         trainTime = str(round((time() - start) / 60, ndigits=2))
#
#         if verbose > 0:
#             print(f"[INFO:] Computation time for training the full classification FNN: {trainTime} min")
#         # plot_history_vis(hist,
#         #                model_hist_plot_path.replace("Fold-" + str(fold_no), "full.DNN-model"),
#         #                model_hist_csv_path.replace("Fold-" + str(fold_no), "full.DNN-model"),
#         #                model_hist_plot_path_a.replace("Fold-" + str(fold_no), "full.DNN-model"),
#         #                model_hist_plot_path_l.replace("Fold-" + str(fold_no), "full.DNN-model"), target)
#         # print(f'[INFO]: Full model for DNN is saved:\n        - {fullModelfile}')
#
#         pd.DataFrame(hist.history).to_csv(fullModelfile.replace(".hdf5", ".history.csv"))
#
#         del model
#         # now next target
#

# ------------------------------------------------------------------------------------- #

def smilesSet2fpSet(csvfilename, outfilename, fptype):
    """

    :param csvfilename: csv file containing a column named 'smiles'
    :param fptype:
    :return: void
    """
    # csvfilename="/data/bioinf/projects/data/2019_Sun-etal_Supplement/results/05_04_dataKS.csv"
    # outfilename = "/data/bioinf/projects/data/2019_Sun-etal_Supplement/results/05_04_dataKS.fp.csv"

    # read csv and generate/add fingerprints to dict
    with open(csvfilename, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')

        feature = 'smiles'

        with open(outfilename, 'w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=['fp'])
            writer.writeheader()
            for row in reader:
                # smiles, need to be converted to fp first
                fp = smi2fp(smile=row[feature], fptype=fptype)
                writer.writerow({'fp': DataStructs.BitVectToText(fp)})

    return


