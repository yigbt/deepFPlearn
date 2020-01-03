# Python module for deepFPlearn tools
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from keras.layers import Dense, Dropout
from keras import regularizers
from keras import optimizers
from keras.models import model_from_json

# for model prediction metrics
import sklearn
from sklearn.metrics import confusion_matrix



# ------------------------------------------------------------------------------------- #

def smi2fp(smile, fptype):
    """
    Convert a SMILES string to a fingerprint object of a certain type using functions
    from the RDKIT python library.

    :param smile: A single SMILES string
    :param fptype: The type of fingerprint to which the SMILES should be converted. Valid
                   values are: 'topological' (default), 'MACCS'
    :return: A fingerprint object
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
            fp = Chem.RDKFingerprint(mol, fpSize=2048)
        except:
            print('SMILES not convertable to topological fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)

    elif fptype == 'MACCS':
        # MACCS Keys:
        # There is a SMARTS-based implementation of the 166 public MACCS keys.
        # The MACCS keys were critically evaluated and compared to other MACCS
        # implementations in Q3 2008. In cases where the public keys are fully defined,
        # things looked pretty good.

        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
        except:
            print('SMILES not convertable to MACSS fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)

    elif fptype == 'atompairs':
        # Atom Pairs:
        # Atom-pair descriptors [3] are available in several different forms.
        # The standard form is as fingerprint including counts for each bit instead
        # of just zeros and ones. Nevertheless we use the BitVect variant here.

        try:
            fp = Pairs.GetAtomPairFingerprintAsBitVect(mol)
            # counts if features also possible here! needs to be parsed differently
            # fps.update({i:Pairs.GetAtomPairFingerprintAsIntVect(mols[i])})
        except:
            print('SMILES not convertable to atompairs fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)

    else:
        # Topological Torsions:
        # At the time of this writing, topological torsion fingerprints have too
        # many bits to be encodeable using the BitVector machinery, so there is no
        # GetTopologicalTorsionFingerprintAsBitVect function.

        try:
            fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
        except:
            print('SMILES not convertable to torsions fingerprint:')
            assert isinstance(smile, object)
            print('SMILES: ' + smile)

    return fp


# ------------------------------------------------------------------------------------- #

def XfromInput(csvfilename, rtype, fptype, printfp=False, retNames=False):
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
    rows = {}
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
            rows.update({i:row})

            # add rowname or number
            if rnameIDX is None:
                rnames.append(str(i))
            else:
                rnames.append(row['id'])

            #print(rnames[i] + ' ' + row[feature])

            # add fp or smile
            if rtype == 'fp':
                # type == fp, fine - add it
              fps.update({i: row[feature]})
            else:
                # smiles, need to be converted to fp first
                fp=smi2fp(smile=row[feature], fptype=fptype)
                fps.update({i: fp})
            i = i + 1

    # split all fingerprints into vectors
    Nrows = len(fps)
    Ncols = len(DataStructs.BitVectToText(fps[0]))

    # Store all fingerprints in numpy array
    x = np.empty((Nrows, Ncols), int)

    if printfp:
        csvoutfilename=csvfilename.replace(".csv", ".fingerprints.csv")
        fnames=names.copy()
        fnames.append('fp')
        f=open(csvoutfilename, 'w')
        writer=csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

        for i in fps:
            fp = DataStructs.BitVectToText(fps[i])
            rows[i]['fp']=fp
            writer.writerow(rows[i])
            x[i] = list(map(int, [char for char in fp]))

        f.close()
    else:
        for i in fps:
            # get fingerprint as string
            fp=DataStructs.BitVectToText(fps[i])
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

def defineNNmodel(inputSize=2048, l2reg=0.001, dropout=0.2, activation='relu', optimizer='Adam'):
    """
    Define the Keras NN model used for training and prediction.

    :param inputSize: The size of the input layer (1dimensional, equals size of fingerprint)
    :return: A keras NN sequential model that needs to be compiled before usage
    """

    #l2reg = 0.001
    #dropout = 0.2

    model = Sequential()

    # input layer has shape of 'inputSize', its the input to 1st hidden layer

    # hidden layers
    model.add(Dense(units=500, activation=activation, input_dim=inputSize,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    model.add(Dense(units=200, activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    model.add(Dense(units=100, activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    model.add(Dense(units=20, activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))

    # output layer
    model.add(Dense(units=1, activation='sigmoid'))

    # compile model
    model.compile(loss="mse", optimizer=optimizer, metrics=['accuracy'])
#   model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# ------------------------------------------------------------------------------------- #

def defineNNmodel2(l2reg=0.001, dropout=0.2, activation='relu', optimizer='Adam'):

    model = Sequential()

    # input layer has shape of 'inputSize', its the input to 1st hidden layer

    # hidden layers
    model.add(Dense(units=500, activation=activation, #input_dim=2048,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    model.add(Dense(units=200, activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    model.add(Dense(units=100, activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    model.add(Dense(units=20, activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))

    # output layer
    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# ------------------------------------------------------------------------------------- #

def predictValues(modelfilepath, pdx):
    """
    Predict a set of chemicals using a selected model.

    :param modelfilepath: Path prefix to the two model files (without .weights.h5 and .csv)
    :param pdx: A matrix containing the fingerprints of the chemicals, generated via XfromInput function
    :return: A dataframe of 2 columns: random - predicted values using random model, trained - predicted values
    using trained model. Rownames are consecutive numbers of the input rows, or if provided in the input file
    the values of the id column
    """
    mfpW = modelfilepath + '.weights.h5'
    mfpM = modelfilepath + '.json'

    print(modelfilepath)
    print(mfpW)
    print(mfpM)

    # learning rate
    lr = 0.001
    # type of optimizer
    adam = optimizers.Adam(lr=lr)

    x = pdx.loc[pdx.index[:],:].to_numpy()

    # random model
    modelRandom = defineNNmodel(inputSize=x.shape[1])
    modelRandom.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
    # predict with random model
    pR = modelRandom.predict(x)

    # trained model
    json_file = open(mfpM, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    modelTrained = model_from_json(loaded_model_json)
    modelTrained.load_weights(mfpW)
    modelTrained.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
    # predict with trained model
    pT = modelTrained.predict(x)

    predictions = pd.DataFrame(data={'random':pR.flatten(),
                                     'trained':pT.flatten()},
                               columns=['random','trained'],
                               index=pdx.index)

    return predictions


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
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy - ' + target)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
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
