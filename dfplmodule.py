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
#matplotlib.use('Agg')
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
#%matplotlib inline
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

    return (df1.iloc[:, 0:x.shape[1]], df1.iloc[:,x.shape[1]:])

    #return gather(df0, key="target", value="association",
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
    mol = Chem.MolFromSmiles(smile)

    # init fp, any better idea? e.g. calling a constructor?
    fp = Chem.Mol #FingerprintMols.FingerprintMol(mol)

    if fptype == 'topological':  # 2048 bits
        # Topological Fingerprints:
        # The fingerprinting algorithm used is similar to that used in the Daylight
        # fingerprinter: it identifies and hashes topological paths (e.g. along bonds)
        # in the molecule and then uses them to set bits in a fingerprint of user-specified
        # lengths. After all paths have been identified, the fingerprint is typically
        # folded down until a particular density of set bits is obtained.
        try:
            #fp = Chem.RDKFingerprint(mol, fpSize=size)
            return(Chem.RDKFingerprint(mol, fpSize=size))
        except:
            print('SMILES not convertable to topological fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)
            return(None)

    elif fptype == 'MACCS':
        # MACCS Keys:
        # There is a SMARTS-based implementation of the 166 public MACCS keys.
        # The MACCS keys were critically evaluated and compared to other MACCS
        # implementations in Q3 2008. In cases where the public keys are fully defined,
        # things looked pretty good.

        try:
            #fp = MACCSkeys.GenMACCSKeys(mol)
            return(MACCSkeys.GenMACCSKeys(mol))
        except:
            print('SMILES not convertable to MACSS fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)
            return(None)

    elif fptype == 'atompairs':
        # Atom Pairs:
        # Atom-pair descriptors [3] are available in several different forms.
        # The standard form is as fingerprint including counts for each bit instead
        # of just zeros and ones. Nevertheless we use the BitVect variant here.

        try:
            #fp = Pairs.GetAtomPairFingerprintAsBitVect(mol)
            return(Pairs.GetAtomPairFingerprintAsBitVect(mol))
            # counts if features also possible here! needs to be parsed differently
            # fps.update({i:Pairs.GetAtomPairFingerprintAsIntVect(mols[i])})
        except:
            print('SMILES not convertable to atompairs fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)
            return(None)

    else:
        # Topological Torsions:
        # At the time of this writing, topological torsion fingerprints have too
        # many bits to be encodeable using the BitVector machinery, so there is no
        # GetTopologicalTorsionFingerprintAsBitVect function.

        try:
            #fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
            return(Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol))
        except:
            print('SMILES not convertable to torsions fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)
            return(None)

# ------------------------------------------------------------------------------------- #

def XfromInput(csvfilename, rtype, fptype, printfp=False, retNames=False, size=2048, verbose=2):
    """
    Return the matrix of features for training and testing NN models (X) as numpy array.
    Provided SMILES are transformed to fingerprints, fingerprint strings are then split
    into vectors and added as row to the array which is returned.

    :param csvfilename: Filename of CSV files containing the training data. The
    SMILES/Fingerprints are stored 1st column
    :param rtype: Type of structure represetation. Valid values are: 'fp' and 'smile'
    :param fptype: Type of fingerprint to be generated out
    :param printfp: Print generated fingerprints to file, namely the input file with the
    file ending '.fingerprints.csv'. Default:False
    :return: A pandas dataframe containing the X matrix for training a NN model,
    rownames/numbers of rows, colnames are the positions of the fp vector.
    """

    # dict to store the fingerprints
    fps = {}
    rows = {} # remove this from memory!
    rnames = []

    # read csv and generate/add fingerprints to dict
    with open(csvfilename, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        names = reader.fieldnames
        #print(names)
        feature = names[names.index(rtype)]  # rtype column ('smiles' or 'fp')
        if 'id' in names:
            rnameIDX = names[names.index('id')]
        else:
            rnameIDX = None

        i = 0

        for row in reader:
            #if i==5:
            #    break

            #print(f'i={i}: {row}')
            if printfp:
                rows.update({i:row})
            #print(rnames[i] + ' ' + row[feature])

            # add fp or smile
            if rtype == 'fp':
                # type == fp, fine - add it
                # add rowname or number
                if rnameIDX is None:
                    rnames.append(str(i))
                else:
                    rnames.append(row['id'])
                # add fingerprint to dictionay
                fps.update({i: row[feature]})
                i = i + 1
            else:
                # smiles, need to be converted to fp first
                fp=smi2fp(smile=row[feature], fptype=fptype, size=size).ToBitString()
                if fp:
                    # add rowname or number
                    if rnameIDX is None:
                        rnames.append(str(i))
                    else:
                        rnames.append(row['id'])

                    # add fingerprint to dictionay
                    fps.update({i: fp})
                    i = i + 1
            #print(f' .. Row done.\n')

    # split all fingerprints into vectors
    Nrows = len(fps)
    Ncols = len(fps[0])
    #Ncols = len(DataStructs.BitVectToText(fps[0]))
    if verbose > 0:
        print(f'[INFO] Returned # of fingerprints: {Nrows}')

    # Store all fingerprints in numpy array
    x = np.empty((Nrows, Ncols), int)

    if printfp:
        csvoutfilename=csvfilename.replace(".csv", "." + str(size) + ".fingerprints.csv")
        fnames=names.copy()
        fnames.append('fp')
        f=open(csvoutfilename, 'w')
        writer=csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

        for i in fps:
            #fp=DataStructs.BitVectToText(fps[i])
            fp = fps[i]
            rows[i]['fp']=fp
            writer.writerow(rows[i])
            x[i] = list(map(int, [char for char in fp]))

        f.close()
        del rows
    else:
        for i in fps:
            # get fingerprint as string
            #fp=DataStructs.BitVectToText(fps[i])
            fp = fps[i]
            # split fp into list of integers
            x[i] = list(map(int, [char for char in fp]))

    pdx = pd.DataFrame(data=x, index=rnames)

    return pdx


# ------------------------------------------------------------------------------------- #

def YfromInput(csvfilename):
    """
    Extract the matrix of outcomes for training/testing NN models that belongs to the
    feature matrix.

    :param csvfilename: Filename of comma separated CSV files containing the training data.
    Target associations start in column 2nd column
    :return: A pandas dataframe containing the Y matrix for training a NN model including
    the names of the targets (each column is a different target)
    """

    df = pd.read_csv(csvfilename)
    y = df[df.columns[1:]]

    return y

# ------------------------------------------------------------------------------------- #

def TrainingDataHeatmap(x, y):

    x['ID'] = x.index
    y['ID'] = y.index

    #xy = pd.merge(x,y,on="ID")

    # clustermap dies because of too many iterations..
    #sns.clustermap(x, metric="correlation", method="single", cmap="Blues", standard_scale=1) #, row_colors=row_colors)

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

def defineCallbacks(checkpointpath, patience, rlrop=False, rlropfactor=0.1, rlroppatience=50):
    """

    :param checkpointpath:
    :param patience:
    :param rlrop:
    :param rlropfactor:
    :param rlroppatience:
    :return:
    """

    # enable this checkpoint to restore the weights of the best performing model
    checkpoint = ModelCheckpoint(checkpointpath, monitor='val_loss', verbose=1,
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
    return(callbacks)

# ------------------------------------------------------------------------------------- #

def defineNNmodelMulti(inputSize=2048, outputSize=None, l2reg=0.001, dropout=0.2,
                       activation='relu', optimizer='Adam', lr=0.001, decay=0.01):

    if optimizer=='Adam':
        myoptimizer = optimizers.Adam(learning_rate=lr, decay=decay)#, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer=='SGD':
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

def defineNNmodel(inputSize=2048, l2reg=0.001, dropout=0.2, activation='relu', optimizer='Adam', lr=0.001, decay=0.01):
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

    if optimizer=='Adam':
        myoptimizer = optimizers.Adam(learning_rate=lr, decay=decay)#, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer=='SGD':
        myoptimizer = SGD(lr=lr, momentum=0.9, decay=decay)
    else:
        myoptimizer = optimizer

    myhiddenlayers = {"2048":6, "1024":5, "999":5, "512":4, "256":3}

    if not str(inputSize) in myhiddenlayers.keys():
        print("Wrong inputsize. Must be in {2048, 1024, 999, 512, 256}.")
        return None

    nhl = int(math.log2(inputSize)/2 - 1)

    model = Sequential()
    # From input to 1st hidden layer
    model.add(Dense(units=int(inputSize/2), input_dim=inputSize,
                    activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    # next hidden layers
    for i in range(1,nhl):
        factorunits = 2**(i+1)
        factordropout = 2*i
        model.add(Dense(units=int(inputSize/factorunits),
                    activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
        model.add(Dropout(dropout/factordropout))
    #output layer
    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()

    # compile model
    model.compile(loss="mse", optimizer=myoptimizer, metrics=['accuracy'])

    return model

# ------------------------------------------------------------------------------------- #

def autoencoderModel(input_size=2048, encoding_dim=256, myactivation='relu', myloss='binary_crossentropy', myregularization=10-3, mylr=0.001, mydecay=0.01):
    """
    This function provides an autoencoder model to reduce a certain input to a compressed version.

    :param encoding_dim: Size of the compressed representation. Default: 85
    :param input_size: Size of the input. Default: 2048
    :param myactivation: Activation function, see Keras activation functions for potential values. Default: relu
    :param myoptimizer: Optimizer, see Keras optmizers for potential values. Default: adadelta
    :param myloss: Loss function, see Keras Loss functions for potential values. Default: binary_crossentropy
    :return: a tuple of autoencoder, encoder and decoder models
    """

    myoptimizer = optimizers.Adam(learning_rate=mylr, decay=mydecay)  # , beta_1=0.9, beta_2=0.999, amsgrad=False)

    # get the number of meaningful hidden layers (latent space included)
    nhl = round(math.log2(input_size/encoding_dim))

    # the input placeholder
    input_vec = Input(shape=(input_size,))

    # 1st hidden layer, that receives weights from input layer
    # equals bottle neck layer, if nhl==1!
    encoded = Dense(units=int(input_size/2), activation='relu')(input_vec)

    if nhl > 1:
        # encoding layers, incl. bottle neck
        for i in range(1, nhl):
            factorunits = 2 ** (i + 1)
            #print(f'{factorunits}: {int(input_size / factorunits)}')
            encoded = Dense(units=int(input_size / factorunits), activation='relu')(encoded)

#        encoding_dim = int(input_size/factorunits)

        # 1st decoding layer
        factorunits = 2 ** (nhl - 1)
        decoded = Dense(units=int(input_size/factorunits), activation='relu')(encoded)

        # decoding layers
        for i in range(nhl-2, 0, -1):
            factorunits = 2 ** i
            #print(f'{factorunits}: {int(input_size/factorunits)}')
            decoded = Dense(units=int(input_size/factorunits), activation='relu')(decoded)

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

def predictValues(acmodelfilepath, modelfilepath, pdx):
    """
    Predict a set of chemicals using a selected model.

    :param modelfilepath: Path to the model weights of the prediction DNN
    :param pdx: A matrix containing the fingerprints of the chemicals, generated via XfromInput function
    :return: A dataframe of 2 columns: random - predicted values using random model, trained - predicted values
    using trained model. Rownames are consecutive numbers of the input rows, or if provided in the input file
    the values of the id column
    """

    #acmodelfilepath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/model.2048.256.ER.checkpoint.AC-model.hdf5"
    #modelfilepath  ="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/model.2048.256.ER.checkpoint.model.hdf5"

    print(f"[INFO:] Loaded model weights for autoencoder: '{modelfilepath}'")
    print(f"[INFO:] Loaded model weights for prediction DNN: '{modelfilepath}'")

    start = time()
    # create autoencoder
    (autoencoder, encoder) = autoencoderModel(input_size=pdx.shape[1], encoding_dim=256)
    # load AC weights
    autoencoder.load_weights(acmodelfilepath)
    # encode input
    encodedX = encoder.predict(pdx)

    trainTime = str(round((time() - start) / 60, ndigits=2))
    print(f"[INFO:] Computation time used for encoding: {trainTime} min")

    start = time()
    # create DNN
    predictor = defineNNmodel(inputSize=encodedX.shape[1])
    # predict with random weights
    predictions_random = predictor.predict(encodedX)
    # load DNN weights
    predictor.load_weights(modelfilepath)
    # predict encoded input
    predictions = predictor.predict(encodedX)

    trainTime = str(round((time() - start) / 60, ndigits=2))
    print(f"[INFO:] Computation time used for the predictions: {trainTime} min")

    df = pd.DataFrame(data={'random':predictions_random.flatten(),
                            'trained':predictions.flatten()},
                      columns=['random','trained'],
                      index=pdx.index)
    print(df)
    return(df)

# ------------------------------------------------------------------------------------- #

#def trainAutoencoder(checkpointpath, X_train, X_test, y_train, y_test, epochs, enc_dim, split, verbose, kfold):
def trainAutoencoder(checkpointpath, Xtrain, Xtest, epochs, enc_dim, verbose, fold):
    """

    :param checkpointpath:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param epochs:
    :return:
    """

    #checkpointpath = checkpointpath.replace(".hdf5", "_Fold-" + str(fold) + ".hdf5")

    # collect the callbacks for training
    callback_list = defineCallbacks(checkpointpath=checkpointpath, patience=20, rlrop=False)
    # Set up the model of the AC w.r.t. the input size and the dimension of the bottle neck (z!)
    (autoencoder, encoder) = autoencoderModel(input_size=Xtrain.shape[1], encoding_dim=enc_dim)

    # Fit the AC
    autohist = autoencoder.fit(Xtrain, Xtrain,
                               callbacks=callback_list,
                               epochs=epochs, batch_size=128, shuffle=True, verbose=verbose,
                               validation_data=(Xtest, Xtest))
    # history
    ac_loss = autohist.history['loss']
    ac_val_loss = autohist.history['val_loss']
    ac_epochs = range(ac_loss.__len__())
    # generate a figure of the losses for this fold
    plt.figure()
    plt.plot(ac_epochs, ac_loss, 'bo', label='Training loss')
    plt.plot(ac_epochs, ac_val_loss, 'b', label='Validation loss')
    plt.title(f'Training and validation loss of AC (Fold = {fold}')
    plt.legend()
    plt.savefig(fname=checkpointpath.replace(".hdf5",
                                             "_trainValLoss_AC_Fold-" + str(fold) + ".svg"),
                format='svg')
    plt.close()
    # write the losses to .csv file for later data visualization
    df = pd.DataFrame(data={'loss': ac_loss, 'val_loss': ac_val_loss, 'epoch': ac_epochs})
    df.to_csv(checkpointpath.replace(".hdf5",
                                     "_trainValLoss_AC_Fold-" + str(fold) + ".csv"))

    # model needs to be saved and restored when predicting new input!
    # use encode() of train data as input for DL model to associate to chemical
    return (encoder.predict(Xtrain), encoder.predict(Xtest), ac_loss.pop(), ac_val_loss.pop())

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

def smooth_curve(points, factor=0.75):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# 02 ---------------------------------------------------------------------------------- #

def set_plot_history_data(ax, history, which_graph):

    if which_graph == 'acc':
        train = smooth_curve(history.history['accuracy'])
        valid = smooth_curve(history.history['val_accuracy'])

    if which_graph == 'loss':
        train = smooth_curve(history.history['loss'])
        valid = smooth_curve(history.history['val_loss'])

    #plt.xkcd() # make plots look like xkcd

    epochs = range(1, len(train) + 1)

    trim = 0 # remove first 5 epochs
    # when graphing loss the first few epochs may skew the (loss) graph

    ax.plot(epochs[trim:], train[trim:], 'dodgerblue', linewidth=15, alpha=0.1)
    ax.plot(epochs[trim:], train[trim:], 'dodgerblue', label=('Training'))

    ax.plot(epochs[trim:], valid[trim:], 'g', linewidth=15, alpha=0.1)
    ax.plot(epochs[trim:], valid[trim:], 'g', label=('Validation'))

# 03 ---------------------------------------------------------------------------------- #

def get_max_validation_accuracy(history):
    validation = smooth_curve(history.history['val_accuracy'])
    ymax = max(validation)
    return 'Max validation accuracy ≈ ' + str(round(ymax, 3)*100) + '%'

def get_max_training_accuracy(history):
    training = smooth_curve(history.history['accuracy'])
    ymax = max(training)
    return 'Max training accuracy ≈ ' + str(round(ymax, 3)*100) + '%'

# 04---------------------------------------------------------------------------------- #

def plot_history(history, file):
    fig, (ax1, ax2) = plt.subplots(nrows=2,
                                   ncols=1,
                                   figsize=(10, 6),
                                   sharex=True,
                                   gridspec_kw={'height_ratios': [5, 2]})

    set_plot_history_data(ax1, history, 'acc')

    set_plot_history_data(ax2, history, 'loss')

    # Accuracy graph
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(bottom=0.5, top=1)
    ax1.legend(loc="lower right")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    ax1.spines['bottom'].set_visible(False)

    # max accuracy text
    plt.text(0.5,
             0.6,
             get_max_validation_accuracy(history),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax1.transAxes,
             fontsize=12)
    plt.text(0.5,
             0.8,
             get_max_training_accuracy(history),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax1.transAxes,
             fontsize=12)

    # Loss graph
    ax2.set_ylabel('Loss')
    ax2.set_yticks([])
    ax2.plot(legend=False)
    ax2.set_xlabel('Epochs')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(fname=file, format='svg')
    plt.close()



# ------------------------------------------------------------------------------------- #

def drawHeatmap(data, anno):

    #(data=pd.DataFrame(Xt), anno = pd.DataFrame(Yt.astype(int)))

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
    #sns.clustermap(df)

    # Clustermethods
    my_palette = dict()
    sns.clustermap(df, metric="correlation", standard_scale=1, method="single", cmap="Blues", row_colors=row_colors)


###############################################################################
# TRAIN FUNCTIONS --------------------------------------------------------- #

def parseInputTrain(parser):
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
#    parser = argparse.ArgumentParser(description='Train a DNN to associate chemical fingerprints with a (set of)'
#                                     'target(s). Trained models are saved to disk including fitted weights and '
#                                     'can be used in the deepFPlearn-Predict.py tool to make predictions.')
    parser.add_argument('-i', metavar='FILE', type=str,
                        help="The file containin the data for training in (unquoted) "
                             "comma separated CSV format. First column contain the feature string in "
                             "form of a fingerprint or a SMILES (see -t option). "
                             "The remaining columns contain the outcome(s) (Y matrix). "
                             "A header is expected and respective column names are used "
                             "to refer to outcome(s) (target(s)).", required=True)
    parser.add_argument('-o', metavar='FILE', type=str,
                        help='Prefix of output file name. Trained model(s) and '
                             'respective stats will be returned in 2 output files '
                             'with this prefix. Default: prefix of input file name.')
    parser.add_argument('-t', metavar='STR', type=str, choices=['fp', 'smiles'],
                        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
                        required=True)
    parser.add_argument('-k', metavar='STR', type=str,
                        choices=['topological', 'MACCS'],  # , 'atompairs', 'torsions'],
                        help='The type of fingerprint to be generated/used in input file.',
                        default=['topological'])
    parser.add_argument('-s', type=int,
                        help = 'Size of fingerprint that should be generated.',
                        default=2048)
    parser.add_argument('-a', action='store_true',
                        help='Use autoencoder to reduce dimensionality of fingerprint. Default: not set.')
    parser.add_argument('-d', metavar='INT', type=int,
                        help='Size of encoded fingerprint (z-layer of autoencoder).',
                        default=256)
    parser.add_argument('-e', metavar='INT', type=int,
                        help='Number of epochs that should be trained',
                        default=50)
    parser.add_argument('-p', metavar='FILE', type=str,
                        help="CSV file containing the parameters for the epochs per target model."
                             "The target abbreviation should be the same as in the input file and"
                             "the columns/parameters are: \n\ttarget,batch_size,epochs,optimizer,activation."
                             "Note that values in from file overwrite -e option!")
    parser.add_argument('-m', action='store_true',
                        help='Train multi-label classification model in addition to the individual models.')
    parser.add_argument('-l', metavar='INT', type=float,
                        help='Fraction of the dataset that should be used for testing. Value in [0,1].',
                        default=0.2)
    parser.add_argument('-K', metavar='INT', type=int,
                        help='K that is used for K-fold cross validation in the training procedure.',
                        default=5)
    parser.add_argument('-v', metavar='INT', type=int, choices=[0,1, 2],
                        help='Verbosity level. O: No additional output, 1: Some additional output, 2: full additional output',
                        default=2
                        )

#    return parser.parse_args()
#return parser

# ------------------------------------------------------------------------------------- #

def defineOutfileNames(pathprefix, mtype, target, fold):
    """
    This function returns the required paths for output files or directories.

    :param pathprefix: A file path prefix for all files.
    :param mtype: The model type. Its set by the trainNNmodels function with information on autoencoder or not,
    and if AC is used, then with its parameters.
    :param target: The name of the target.

    :return: A tuple of 14 output file names.
    """
    
    modelname = mtype + '.' + target + '.Fold-' + str(fold)

    modelfilepathW = str(pathprefix) + '/model.' + modelname + '.weights.h5'
    modelfilepathM = str(pathprefix) + '/model.' + modelname + '.json'
    modelhistplotpathL = str(pathprefix) + '/model.' + modelname + '.loss.svg'
    modelhistplotpathA = str(pathprefix) + '/model.' + modelname + '.acc.svg'
    modelhistplotpath = str(pathprefix) + '/model.' + modelname + '.history.svg'
    modelhistcsvpath = str(pathprefix) + '/model.' + modelname + '.history.csv'
    modelvalidation = str(pathprefix) + '/model.' + modelname + '.validation.csv'
    modelAUCfile = str(pathprefix) + '/model.' + modelname + '.auc.svg'
    modelAUCfiledata = str(pathprefix) + '/model.' + modelname + '.auc.data.csv'
    outfilepath = str(pathprefix) + '/model.' + modelname + '.trainingResults.txt'
    checkpointpath = str(pathprefix) + '/model.' + modelname + '.checkpoint.model.hdf5'
    checkpointpathAC = str(pathprefix) + '/model.' + modelname + '.checkpoint.AC-model.hdf5'
    modelheatmapX = str(pathprefix) + '/model.' + modelname + '.heatmap.X.svg'
    modelheatmapZ = str(pathprefix) + '/model.' + modelname + '.AC.heatmap.Z.svg'

    return (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
            modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
            modelAUCfiledata, outfilepath, checkpointpath, checkpointpathAC,
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

def prepareDataSet(y, x, t):
    """
    A function to remove NA values from the output column, duplicates from input and output and to transform the
    data into numpy arrays for keras functions.
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

    #return(shuffleDataPriorToTraining(x=XtU, y=YtU))
    return(XtU, YtU)

# ------------------------------------------------------------------------------------- #

def plotHistoryVis(hist, modelhistplotpath, modelhistcsvpath, modelhistplotpathA, modelhistplotpathL, target):
    plot_history(history=hist, file=modelhistplotpath)
    histDF = pd.DataFrame(hist.history)
    histDF.to_csv(modelhistcsvpath)

    # plot accuracy and loss for the training and validation during training
    plotTrainHistory(hist=hist, target=target, fileAccuracy=modelhistplotpathA, fileLoss=modelhistplotpathL)

# ------------------------------------------------------------------------------------- #

def validateMultiModelOnTestData(Z_test, checkpointpath, y_test, colnames, resultfile):

    # load checkpoint model with min(val_loss)
    trainedmodel = defineNNmodelMulti(inputSize=Z_test.shape[1], outputSize=y_test.shape[1])

    # predict values with random model
    predictions_random = pd.DataFrame(trainedmodel.predict(Z_test), columns=colnames+'-predRandom')

    # load weights into random model
    trainedmodel.load_weights(checkpointpath)

    # predict with trained model
    predictions = pd.DataFrame(trainedmodel.predict(Z_test),
                                      columns=colnames+'-pred')
    scores = pd.DataFrame((predictions.round() == y_test).sum() / y_test.shape[0], columns=['correctPredictions'])

    results = pd.concat([predictions_random, predictions, pd.DataFrame(y_test, columns=colnames + '-true')], axis=1)
    results.to_csv(resultfile)

    return scores

# ------------------------------------------------------------------------------------- #

def validateModelOnTestData(Z_test, checkpointpath, y_test, modeltype, modelvalidation, target,
                            modelAUCfiledata, modelAUCfile):
    """
    Function that validates trained model with test data set. History and AUC plots are generated.
    Accuracy and Loss of model on test data, as well as MCC and confusion matrix is calculated and returned.

    :param Z_test:
    :param checkpointpath:
    :param y_test:
    :param modeltype:
    :param modelvalidation:
    :param target:
    :param modelAUCfiledata:
    :param modelAUCfile:
    :return: Tupel containing Loss, Accuracy, MCC, tn, fp, fn, tp of trained model on test data.
    """

    # load checkpoint model with min(val_loss)
    trainedmodel = defineNNmodel(inputSize=Z_test.shape[1])

    # predict values with random model
    predictions_random = pd.DataFrame(trainedmodel.predict(Z_test))

    # load weights into random model
    trainedmodel.load_weights(checkpointpath)
    # predict with trained model
    predictions = pd.DataFrame(trainedmodel.predict(Z_test))

    # save validation data to .csv file
    validation = pd.DataFrame({'predicted': predictions[0].ravel(),
                               'true': list(y_test),
                               'predicted_random': predictions_random[0].ravel(),
                               'modeltype': modeltype})
    validation.to_csv(modelvalidation)

    # compute MCC
    predictionsInt = [int(round(x)) for x in predictions[0].ravel()]
    ytrueInt = [int(y) for y in y_test]
    MCC = matthews_corrcoef(ytrueInt,predictionsInt)

    # generate the AUC-ROC curve data from the validation data
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, predictions, drop_intermediate=False)

    auc_keras = auc(fpr_keras, tpr_keras)

    aucdata = pd.DataFrame(list(zip(fpr_keras,
                                    tpr_keras,
                                    [auc_keras for x in range(1, len(fpr_keras))],
                                    [target for x in range(1, len(fpr_keras))])),
                           columns=['fpr', 'tpr', 'auc', 'target'])
    aucdata.to_csv(modelAUCfiledata)

    plotAUC(fpr=fpr_keras, tpr=tpr_keras, target=target, auc=auc_keras, filename=modelAUCfile)

    # [[tn, fp]
    #  [fn, tp]]
    cfm = confusion_matrix(y_true=ytrueInt, y_pred=predictionsInt)

    scores = trainedmodel.evaluate(Z_test, y_test, verbose=0)

    print(f'TARGET: {target} Loss: {scores[0].__round__(2)} Acc: {scores[1].__round__(2)}')
    print(f'MCC: {MCC.__round__(2)}')
    print(f'CFM: \n\tTN={cfm[0][0]}\tFP={cfm[0][1]}\n\tFN={cfm[1][0]}\tTP={cfm[1][1]}')

    return (scores[0], scores[1], MCC, cfm[0][0], cfm[0][1], cfm[1][0], cfm[1][1])

# ------------------------------------------------------------------------------------- #
def trainNNmodelsMulti(modelfilepathprefix, x, y, split=0.2, epochs=500, params=None, enc_dim=256, autoenc=False, verbose=2, kfold=5):

    if autoenc:
        modeltype = str(size) + '.' + str(enc_dim)
    else:
        modeltype = str(size) + '.noAC'

    # drop compounds that are not measured for all target columns, transform to numpy
    (xmulti, ymulti) = shuffleDataPriorToTraining(x, y)

    # do a kfold cross validation for the autoencoder training
    kfoldCValidator = KFold(n_splits=kfold, shuffle=True, random_state=42)

    # store acc and loss for each fold
    allscores = pd.DataFrame(columns=["fold_no",  # fold number of k-fold CV
                                      "loss_AC", "val_loss_AC",  # AC
                                      "loss", "val_loss", "acc", "val_acc",  # FNN training
                                      "loss_test", "acc_test", "mcc_test"])  # FNN test data

    fold_no = 1

    # split the data
    for train, test in kfoldCValidator.split(xmulti, ymulti):

        # define all the output file/path names
        (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
         modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
         modelAUCfiledata, outfilepath, checkpointpath, checkpointpathAC,
         modelheatmapX, modelheatmapZ) = defineOutfileNames(pathprefix=modelfilepathprefix, mtype=modeltype, target="multi")

        # generate train and test sets
#        (X_train, X_test, y_train, y_test) = train_test_split(xmulti.to_numpy(), ymulti.to_numpy(),
#                                                              test_size=(1 - split), random_state=0)

       # use autoencoder for dimensionality reduction of the X matrix
        if autoenc:
            (Z_train, Z_test, ac_loss, ac_val_loss) = trainAutoencoder(checkpointpath=checkpointpathAC,
                                                                       Xtrain=xmulti[train], Xtest=xmulti[test],
                                                                       epochs=epochs,
                                                                       enc_dim=enc_dim,
                                                                       verbose=verbose,
                                                                       fold=fold_no)
        else:
            (Z_train, Z_test, ac_loss, ac_val_loss) = (xmulti[train], xmulti[test], -1, -1)

        # use a dnn for multi-class prediction
        model = defineNNmodelMulti(inputSize=Z_train.shape[1], outputSize=ymulti.shape[1])  # X_train.shape[1])

        callback_list = defineCallbacks(checkpointpath=checkpointpath, patience=20,
                                        rlrop=True, rlropfactor=0.1, rlroppatience=100)
        # measure the training time
        start = time()

        # train and validate
        hist = model.fit(Z_train, ymulti[train],
                         callbacks=callback_list,
                         epochs=epochs, batch_size=128, verbose=2, validation_split=split)

        trainTime = str(round((time() - start) / 60, ndigits=2))

        if verbose > 0:
            print(f"[INFO:] Computation time for training the classification DNN: {trainTime} min")

        # validate model on test data set (x_test, y_test)
        scores = validateMultiModelOnTestData(Z_test=Z_test, checkpointpath=checkpointpath, y_test=ymulti[test],
                                              colnames=y.columns,
                                              resultfile=outfilepath.replace("trainingResults.txt",
                                                                             "predictionResults.csv"))

        idx = hist.history['val_loss'].index(min(hist.history['val_loss']))
        row_df = pd.DataFrame([[fold_no,
                                ac_loss, ac_val_loss,
                                hist.history['loss'][idx], hist.history['val_loss'][idx],
                                hist.history['accuracy'][idx], hist.history['val_accuracy'][idx],
                                scores[0], scores[1], scores[2]]],
                              columns=["fold_no",  # fold number of k-fold CV
                                       "loss_AC", "val_loss_AC",  # AC
                                       "loss", "val_loss", "acc", "val_acc",  # FNN training
                                       "loss_test", "acc_test", "mcc_test"]
                              )
        print(row_df)
        allscores = allscores.append(row_df, ignore_index=True)

        # validate on test data set
        scores = validateMultiModelOnTestData(Z_test=Z_test, checkpointpath=checkpointpath, y_test=ymulti[test],
                                              colnames=y.columns, resultfile=outfilepath.replace("trainingResults.txt",
                                                                                                 "predictionResults.csv"))
        fold_no = fold_no + 1

    print(allscores)

    # finalize model
    # 1. provide best performing fold variant
    # select best model based on MCC
    idx2 = allscores[['mcc_test']].idxmax().ravel()[0]
    fold_no = allscores._get_value(idx2, 'fold_no')
    # get all the filenames
    (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
     modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
     modelAUCfiledata, outfilepath, checkpointpath, checkpointpathAC,
     modelheatmapX, modelheatmapZ) = defineOutfileNames(pathprefix=modelfilepathprefix, mtype=modeltype,
                                                        target="multi", fold=fold_no)

    file = re.sub("\.hdf5", "scores.csv", re.sub("Fold-.\.checkpoint", "Fold-All", checkpointpath))
    allscores.to_csv(file)

    bestModelfileAC = checkpointpathAC.replace("Fold-" + str(fold_no) + ".checkpoint", "best")
    bestModelfile = checkpointpath.replace("Fold-" + str(fold_no) + ".checkpoint.", "best.DNN-")
    # copy best AC model
    shutil.copyfile(checkpointpathAC, bestModelfileAC)
    # copy best DNN model
    shutil.copyfile(checkpointpath, bestModelfile)
    print(f'[INFO]: Best models for AC and DNN are saved:\n        - {bestModelfileAC}\n        - {bestModelfile}')

    # AND retrain with full data set
    fullModelfileAC = checkpointpathAC.replace("Fold-" + str(fold_no) + ".checkpoint", "full")
    fullModelfile = checkpointpath.replace("Fold-" + str(fold_no) + ".checkpoint", "full")
    # measure the training time
    start = time()
    if autoenc:
        (Z_train, Z_test, ac_loss, ac_val_loss) = trainAutoencoder(checkpointpath=fullModelfileAC,
                                                                   Xtrain=xmulti, Xtest=xmulti,
                                                                   epochs=epochs,
                                                                   enc_dim=enc_dim,
                                                                   verbose=verbose,
                                                                   fold=0)  # 0 stands for full retrained model!
        # plotHeatmap(Z_train, filename=modelheatmapZ,title=("Z representation "+target))
    else:
        (Z_train, Z_test, ac_loss, ac_val_loss) = (xmulti, xmulti, -1, -1)

    model = defineNNmodel(inputSize=Z_train.shape[1])  # X_train.shape[1])
    callback_list = defineCallbacks(checkpointpath=fullModelfile, patience=20,
                                    rlrop=True, rlropfactor=0.1, rlroppatience=100)
    # train and validate
    hist = model.fit(Z_train, ymulti,
                     callbacks=callback_list,
                     epochs=epochs, batch_size=128, verbose=2, validation_split=split)
    #                             validation_data=(Z_test, y_test))  # this overwrites val_split!
    trainTime = str(round((time() - start) / 60, ndigits=2))

    if verbose > 0:
        print(f"[INFO:] Computation time for training the classification DNN: {trainTime} min")
    plotHistoryVis(hist,
                   modelhistplotpath.replace("Fold-" + str(fold_no), "full.DNN-model"),
                   modelhistcsvpath.replace("Fold-" + str(fold_no), "full.DNN-model"),
                   modelhistplotpathA.replace("Fold-" + str(fold_no), "full.DNN-model"),
                   modelhistplotpathL.replace("Fold-" + str(fold_no), "full.DNN-model"), target)
    print(f'[INFO]: Full models for AC and DNN are saved:\n        - {fullModelfileAC}\n        - {fullModelfile}')

    pd.DataFrame(hist.history).to_csv(fullModelfile.replace(".hdf5", ".history.csv"))
    # stats.append([target, [x.__round__(2) for x in scores]])


# ------------------------------------------------------------------------------------- #

def trainNNmodels(modelfilepathprefix, x, y, split=0.2, epochs=50, params=None, enc_dim=256, autoenc=False, verbose=2, kfold=5):
    """
    Train individual models for all targets (columns) present in the provided target data (y) and a multi-label
    model that classifies all targets at once. For each individual target the data is first subsetted to exclude NA
    values (for target associations). A random sample of the remaining data (size is the split fraction) is used for
    training and the remaining data for validation.

    :param modelfilepathprefix: A path prefix for all output files
    :param x: The feature matrix.
    :param y: The outcome matrix.
    :param split: The percentage of data used for validation.
    :param epochs: The number of epochs for training the autoencoder and the DNN for classification.
    Note: Early stopping and fallback is enabled.
    :param params: A .csv files containing paramters that should be evaluated. See file tunedParams.csv.
    :param enc_dim: The dimension of latent space vector (z) of the autoencoder.
    :param autoenc: Use the autoencoder.
    :param verbose: Verbosity level.

    :return: A list with loss and accuracy values for each individual model.
    """

    size = x.shape[1]

    stats = []

    if params:
        parameters = pd.read_csv(params)

    # add a target 'summarized' for association to ANY of the target genes
    # maybe it improves the detection of '1's due to increased data set
    mysum = y.sum(axis=1)
    y['summarized'] = [0 if s == 0 else 1 for s in mysum]

    ### For each individual target (+ summarized target)
    for target in y.columns:
        # target=y.columns[1] # --> only for testing the code

        if autoenc:
            modeltype = str(size) + '.' + str(enc_dim)
        else:
            modeltype = str(size) + '.noAC'


        # rm NAs and duplicates, shuffle, and transform to numpy arrays
        (Xt, Yt) = prepareDataSet(y, x, target)

        # do a kfold cross validation for the autoencoder training
        kfoldCValidator = KFold(n_splits=kfold, shuffle=True, random_state=42)

        # store acc and loss for each fold
        allscores = pd.DataFrame(columns=["fold_no",                             # fold number of k-fold CV
                                          "loss_AC", "val_loss_AC",              # AC
                                          "loss", "val_loss", "acc", "val_acc",  # FNN training
                                          "loss_test", "acc_test", "mcc_test"]) # FNN test data

        fold_no = 1

        # split the data
        for train, test in kfoldCValidator.split(Xt, Yt):

        #    print(Xt[train].shape)
        #    print(Xt[test].shape)
        #    continue

            if verbose>0:
                print(f'[INFO]: Training of fold number: {fold_no} ------------------------------------\n')

            # define all the output file/path names
            (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
             modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
             modelAUCfiledata, outfilepath, checkpointpath, checkpointpathAC,
             modelheatmapX, modelheatmapZ) = defineOutfileNames(pathprefix=modelfilepathprefix, mtype=modeltype,
                                                                target=target, fold=fold_no)

            if verbose>0:
               eval01Distributions(Xt=Xt, Yt=Yt, y_train=Yt[train], y_test=Yt[test], verbosity=verbose)

            # Plot a heatmap of the 0-1 values of the feature matrix used for training
            # This should contain mainly zeros for this problem
            #plotHeatmap(Xt[train], filename=modelheatmapX, title=("X representation " + target + "_fold-" + fold_no))

            # Train the autoencoder (AC) to reduce the feature vector for the input layer of the DNN classificator
            if autoenc:
                (Z_train, Z_test, ac_loss, ac_val_loss) = trainAutoencoder(checkpointpath=checkpointpathAC,
                                                                           Xtrain=Xt[train], Xtest=Xt[test],
                                                                           epochs=epochs,
                                                                           enc_dim=enc_dim,
                                                                           verbose=verbose,
                                                                           fold=fold_no)
                #plotHeatmap(Z_train, filename=modelheatmapZ,title=("Z representation "+target))
            else:
                (Z_train, Z_test, ac_loss, ac_val_loss) = (Xt[train], Xt[test], -1, -1)

            # standard parameters are the tuning results
            # if params:
            #     ps = parameters.loc[parameters['target'] == target]
            #     model = defineNNmodel(inputSize=X_train.shape[1], activation=ps['activation'][0], optimizer=ps['optimizer'][0])
            #
            #     start = time()
            #
            #     callback_list = defineCallbacks(checkpointpath=checkpointpath, patience=50,
            #                                          rlrop=True, rlropfactor=0.1, rlroppatience=100)
            #
            #     # train and validate
            #     hist = model.fit(X_train, y_train,
            #                      callbacks=callback_list,
            #                      epochs=ps['epochs'][0],
            #                      batch_size=ps['batch_size'][0], verbose=2,#,
            #                      #validation_split=0.2,
            #                      validation_data=(X_test, y_test)) # this overwrites val_split!
            #     trainTime = str(round((time() - start) / 60, ndigits=2))
            #else:
            model = defineNNmodel(inputSize=Z_train.shape[1])  # X_train.shape[1])

            callback_list = defineCallbacks(checkpointpath=checkpointpath, patience=20,
                                            rlrop=True, rlropfactor=0.1, rlroppatience=100)
            # measure the training time
            start = time()
            # train and validate
            hist = model.fit(Z_train, Yt[train],
                             callbacks=callback_list,
                             epochs=epochs, batch_size=128, verbose=2, validation_split=split)
#                             validation_data=(Z_test, y_test))  # this overwrites val_split!
            trainTime = str(round((time() - start) / 60, ndigits=2))

            if verbose > 0:
                print(f"[INFO:] Computation time for training the classification DNN: {trainTime} min")

                # generate history plots
            #plotHistoryVis(hist, modelhistplotpath, modelhistcsvpath, modelhistplotpathA, modelhistplotpathL, target)

            # validate model on test data set (x_test, y_test)
            scores = validateModelOnTestData(Z_test, checkpointpath, Yt[test], modeltype, modelvalidation, target,
                                             modelAUCfiledata, modelAUCfile)

            idx = hist.history['val_loss'].index(min(hist.history['val_loss']))

            row_df = pd.DataFrame([[fold_no,
                                    ac_loss, ac_val_loss,
                                    hist.history['loss'][idx], hist.history['val_loss'][idx],
                                    hist.history['accuracy'][idx], hist.history['val_accuracy'][idx],
                                    scores[0], scores[1], scores[2]]],
                                  columns=["fold_no",  # fold number of k-fold CV
                                           "loss_AC", "val_loss_AC",  # AC
                                           "loss", "val_loss", "acc", "val_acc",  # FNN training
                                           "loss_test", "acc_test", "mcc_test"]
                                  )
            print(row_df)
            allscores = allscores.append(row_df, ignore_index=True)
            fold_no = fold_no + 1
            # now next fold

        print(allscores)

        # finalize model
        # 1. provide best performing fold variant
        # select best model based on MCC
        idx2 = allscores[['mcc_test']].idxmax().ravel()[0]
        fold_no = allscores._get_value(idx2, 'fold_no')
        # get all the filenames
        (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
         modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
         modelAUCfiledata, outfilepath, checkpointpath, checkpointpathAC,
         modelheatmapX, modelheatmapZ) = defineOutfileNames(pathprefix=modelfilepathprefix, mtype=modeltype,
                                                            target=target, fold=fold_no)

        file = re.sub("\.hdf5", "scores.csv", re.sub("Fold-.\.checkpoint", "Fold-All", checkpointpath))
        allscores.to_csv(file)

        bestModelfileAC = checkpointpathAC.replace("Fold-" + str(fold_no) + ".checkpoint", "best")
        bestModelfile = checkpointpath.replace("Fold-" + str(fold_no) + ".checkpoint.", "best.DNN-")
        # copy best AC model
        shutil.copyfile(checkpointpathAC, bestModelfileAC)
        # copy best DNN model
        shutil.copyfile(checkpointpath, bestModelfile)
        print(f'[INFO]: Best models for AC and DNN are saved:\n        - {bestModelfileAC}\n        - {bestModelfile}')
        # AND retrain with full data set
        fullModelfileAC = checkpointpathAC.replace("Fold-" + str(fold_no) + ".checkpoint", "full")
        fullModelfile = checkpointpath.replace("Fold-" + str(fold_no) + ".checkpoint", "full")
        # measure the training time
        start = time()
        if autoenc:
            (Z_train, Z_test, ac_loss, ac_val_loss) = trainAutoencoder(checkpointpath=fullModelfileAC,
                                                                       Xtrain=Xt, Xtest=Xt,
                                                                       epochs=epochs,
                                                                       enc_dim=enc_dim,
                                                                       verbose=verbose,
                                                                       fold=0) # 0 stands for full retrained model!
            # plotHeatmap(Z_train, filename=modelheatmapZ,title=("Z representation "+target))
        else:
            (Z_train, Z_test, ac_loss, ac_val_loss) = (Xt, Xt, -1, -1)

        model = defineNNmodel(inputSize=Z_train.shape[1])  # X_train.shape[1])
        callback_list = defineCallbacks(checkpointpath=fullModelfile, patience=20,
                                        rlrop=True, rlropfactor=0.1, rlroppatience=100)
        # train and validate
        hist = model.fit(Z_train, Yt,
                         callbacks=callback_list,
                         epochs=epochs, batch_size=128, verbose=2, validation_split=split)
        #                             validation_data=(Z_test, y_test))  # this overwrites val_split!
        trainTime = str(round((time() - start) / 60, ndigits=2))

        if verbose > 0:
            print(f"[INFO:] Computation time for training the classification DNN: {trainTime} min")
        plotHistoryVis(hist,
                       modelhistplotpath.replace("Fold-"+str(fold_no), "full.DNN-model"),
                       modelhistcsvpath.replace("Fold-"+str(fold_no), "full.DNN-model"),
                       modelhistplotpathA.replace("Fold-"+str(fold_no), "full.DNN-model"),
                       modelhistplotpathL.replace("Fold-"+str(fold_no), "full.DNN-model"), target)
        print(f'[INFO]: Full models for AC and DNN are saved:\n        - {fullModelfileAC}\n        - {fullModelfile}')

        pd.DataFrame(hist.history).to_csv(fullModelfile.replace(".hdf5", ".history.csv"))
        #stats.append([target, [x.__round__(2) for x in scores]])

        # now next target

# ------------------------------------------------------------------------------------- #

def trainMultiNNmodel(model, x, y, split=0.8):
    """
    Train one multi-class model of the provided structure for all targets (columns) provided in y
    using the features x and the outcomes y and a train/validation data set split of
    split.
    :param model: a compiled neural network model
    :param x: a numpy array of training features, one row per data set, features in cols.
    :param y: a pandas data frame of training outcomes, one column per outcome.
    Cell value (0,1). Name of column used as name of target.
    :param split: the percentage data sets used for training. Remaining percentage is
    used in cross-validation steps.
    :return: matrix of statistics per target
    """

    stats = {}

    return stats

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
                writer.writerow({'fp':DataStructs.BitVectToText(fp)})

    return

###############################################################################
# PREDICT FUNCTIONS --------------------------------------------------------- #

def parseInputPredict(parser):
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
#    parser = argparse.ArgumentParser(description='Use models that have been generated by deepFPlearn-Train.py '
#                                                 'tool to make predictions on chemicals (provide SMILES or '
#                                                 'topological fingerprints).')
    parser.add_argument('-i', metavar='FILE', type=str,
                        help="The file containin the data for the prediction. It is in"
                             "comma separated CSV format. The column named 'smiles' or 'fp'"
                             "contains the field to be predicted. Please adjust the type "
                             "that should be predicted (fp or smile) with -t option appropriately."
                             "An optional column 'id' is used to assign the outcomes to the"
                             "original identifieres. If this column is missing, the results are"
                             "numbered in the order of their appearance in the input file."
                             "A header is expected and respective column names are used.",
                        required=True)
    parser.add_argument('--ACmodel', metavar='FILE', type=str,
                        help='The autoencoder model weights',
                        required=True)
    parser.add_argument('--model', metavar='FILE', type=str,
                        help='The predictor model weights',
                        required=True)
    parser.add_argument('-o', metavar='FILE', type=str,
                        help='Output file name. It containes a comma separated list of '
                             "predictions for each input row, for all targets. If the file 'id'"
                             "was given in the input, respective IDs are used, otherwise the"
                             "rows of output are numbered and provided in the order of occurence"
                             "in the input file.")
    parser.add_argument('-t', metavar='STR', type=str, choices=['fp', 'smiles'],
                        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
                        required=True)
    parser.add_argument('-k', metavar='STR', type=str,
                        choices=['topological', 'MACCS'],  # , 'atompairs', 'torsions'],
                        help='The type of fingerprint to be generated/used in input file.',
                        default='topological')

#    return parser.parse_args()
