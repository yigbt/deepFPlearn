import argparse
import os
import re
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from time import time

import dfplmodule as dfpl

# ------------------------------------------------------------------------------------- #

def parseInput():
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    parser = argparse.ArgumentParser(description='Perform Hyperparameter tuning for a '
                                     'given set of parameters.')
    # input
    parser.add_argument('-i', metavar='FILE', type=str, nargs=1, required=True,
                        help="The file containin the data for training in (unquoted) "
                             "comma separated CSV format. The first column(s) contain "
                             "the target associations, while the last column, named 'fp' "
                             "contains the fingerprint of the compound")
    # output
    parser.add_argument('-p', metavar='PATH', type=str, nargs=1, required=True,
                        help="The path to a directory where the results of the hp tuning "
                             "shall be stored")
    # select target(s) or use all
    parser.add_argument('-t', metavar='STR', type=str, nargs='+', required=False,
                        help='A space separated list of target names (according to column '
                             ' names in the input file (-i)) for which the huperparameter '
                             'tuning should be performed. If not provided, it is performed '
                             'for each target seperately.')

    # parameters to be tested for HP tuning
    parser.add_argument('--batchSizes', metavar='INT', type=int, nargs='+',
                        required=False, default=[20],
                        help="Space separated list of integers for batch sizes that should "
                             "be tested. Example: --batchSizes 32 64")
    parser.add_argument('--epochs', metavar='INT', type=int, nargs='+',
                        required=False, default=[30],
                        help="Space separated list of integers for number of training epochs "
                             "that should be tested. Example: --epochs 30 50 100")
    parser.add_argument('--inits', metavar='STR', type=str, nargs='+',
                        required=False, default=['normal'],
                        help="Space separated list of methods for initializing the weights "
                             "at inner layers that should be tested. Available methods are: "
                             "'glorot_uniform', 'normal', 'uniform'. "
                             "Example: --inits 'glorot_uniform' 'normal'")
    parser.add_argument('--optimizers', metavar='STR', type=str, nargs='+',
                        required=False, default=['Adam'],
                        help="Space separated list of optimizers for the DNN model "
                             "that should be tested."
                             "Available optimizers are listed here: https://keras.io/optimizers/ ."
                             "Example: --optimizers 'Adadelta' 'Adam' 'Adamax'"
                             )
    parser.add_argument('--activations', metavar='STR', type=str, nargs='+',
                        required=False, default=['relu'],
                        help="Space separated list of activation fucntions for the DNN model "
                             "that should be tested."
                             "Available functions are listed here: https://keras.io/activations/ ."
                             "Example: --activations 'sigmoid' 'relu'"
                             )

    return parser.parse_args()

# ----------------------------------------------------------------------------- #

# model for tuning epochs and batchsizes only
def c_model(dropout=0.2):
    model = Sequential()
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(341, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(85, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(17, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ----------------------------------------------------------------------------- #

# model for tuning optmizer, activation functions and initialization of hidden layers
def tuning_model(optimizer, activation, init, dropout=0.2):
    model = Sequential()
    model.add(Dense(1024, activation=activation, init=init))
    model.add(Dropout(dropout))
    model.add(Dense(341, activation=activation, init=init))
    model.add(Dropout(dropout))
    model.add(Dense(85, activation=activation, init=init))
    model.add(Dropout(dropout))
    model.add(Dense(17, activation=activation, init=init))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# --------------------------------------------------------------------------- #

# ===================================================================================== #

if __name__ == '__main__':

    # get all arguments
    args = parseInput()

    # this stores the best performing parameters for each model for each target
    results = pd.DataFrame()

    np.random.seed(0)

    #filepath = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
    #outfilepath = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/" + re.sub(".csv", ".hpTuningResults.txt", os.path.basename(args.i[0]))
    #dataset = pd.read_csv(filepath)

    outfilepath = args.p[0] + re.sub(".csv", ".hpTuningResults.txt", os.path.basename(args.i[0]))

    dataset = pd.read_csv(args.i[0])

    #le = LabelEncoder()

    ## which target column to use? put this in a loop later for all targets
    ## here we need a for loop

    for target in args.t:
        #target = 'ER'
        print(target)
        if(target in dataset.columns):
            #modelfilepathW = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/" + '/model.' + target + '.weights.h5'
            #modelfilepathM = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/" + '/model.' + target + '.json'
            modelfilepathW = args.p[0] + '/model.' + target + '.weights.h5'
            modelfilepathM = str(modelfilepathprefix) + '/model.' + target + '.json'

            tmp = dataset[target].astype('category')
            Y = np.asarray(tmp)
            naRows = np.isnan(Y)

            d = dataset[~naRows] # d is a copy!

            # generate X from fingerprints
            # split all fingerprints into vectors
            Nrows = d.shape[0]
            Ncols = d['fp'][0].__len__()
            # Store all fingerprints in numpy array
            X = np.empty((Nrows, Ncols), int)

            # keep old indexes of this subset of arrays
            d['oldIdx'] = d.index.values
            d['newIdx'] = range(Nrows)
            d = d.set_index('newIdx')
            del d.index.name

            # generate X matrix
            for i in range(Nrows):
                fp = d['fp'][i]
                X[i] = np.array(list(fp), dtype='int')

            # generate y vector(s)
            y = to_categorical(np.array(d[target]), num_classes=2)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            start = time()

            # Start optimizing epochs and batchsizes (if more than one provided)
            model = KerasClassifier(build_fn=dfpl.defineNNmodel2)

            batchSizes = args.batchSizes # batchSizes = [20]
            epochs = args.epochs # epochs = [5, 10]

            parameters = {'batch_size': batchSizes,
                          'epochs': epochs}

            clf = GridSearchCV(model, parameters, verbose=0)
            clf_results = clf.fit(X_train, y_train)



            inits = ['glorot_uniform', 'normal', 'uniform']
            optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
            activation_functions = ['sigmoid', 'tanh', 'relu']
            parameters = {'batch_size': batch_sizes,
                          'epochs': epochs,
                          'optimizer': optimizers,
                          'activation': activation_functions,
                          'init': inits}

            # Tuning ALL parameters, Note, this takes hours of computation!!!
            clf = GridSearchCV(model, parameters, verbose=0)
            clf_results = clf.fit(X_train, y_train)

            # save best estimator per target
            clf_results.best_estimator_.model.save(filepath=modelfilepathM)
            clf_results.best_estimator_.model.save_weights(filepath=modelfilepathW)

            ### find best performing parameters
            file = open(outfilepath, "a")
            file.write("# --------------------------------------------------------------------------- #\n")
            file.write("### Results for %s target ###\n" % target)
            file.write("Best: %f using %s\n" % (clf_results.best_score_, clf_results.best_params_))
            file.write("Calculation time: %s min\n\n" % str(round((time()-start)/60, ndigits=2)))
            file.close()

        else:
            print("ERROR: the target that you provide (%s) "
                  "is not contained in your data file (%s)" %
                  (target, args.i[0])
