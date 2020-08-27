import sys
import argparse
import os
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from time import time

from dfpl import dfplmodule as dfpl


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

    #print(args)
    #exit(1)

    # this stores the best performing parameters for each model for each target
    results = pd.DataFrame()

    np.random.seed(0)

    #filepath = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
    #outfilepath = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/" + re.sub(".csv", ".hpTuningResults.txt", os.path.basename(filepath))
    #dataset = pd.read_csv(filepath)

    dataset = pd.read_csv(args.i[0])

    #le = LabelEncoder()

    ## which target column to use? put this in a loop later for all targets
    ## here we need a for loop

    for target in args.t:
        target = 'ER'
        print(target)
        #outfilepath = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/" + re.sub(".csv", '.hpTuningResults.' + target + '.txt', os.path.basename(filepath))
        outfilepath = args.p[0] + re.sub(".csv", '.hpTuningResults.' + target + '.txt', os.path.basename(args.i[0]))


        if(target in dataset.columns):
            #modelfilepathW = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/" + '/model.' + target + '.weights.h5'
            #modelfilepathM = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/" + '/model.' + target + '.json'
            modelfilepathW = args.p[0] + '/model.' + target + '.weights.h5'
            modelfilepathM = args.p[0] + '/model.' + target + '.json'

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
            #y = to_categorical(np.array(d[target]), num_classes=2)
            y = d[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            start = time()

            ### find best performing parameters
            sys.stdout.write("# --------------------------------------------------------------------------- #\n")
            sys.stdout.write("#target = %s\n" % target)

            # Start optimizing epochs and batchsizes (if more than one provided)

            #batchSizes = args.batchSizes
            batchSizes = [64, 128, 256]
            epochs = args.epochs  # epochs = [5, 10]

            if (batchSizes.__len__() > 1):

                #model = KerasClassifier(build_fn=dfpl.defineNNmodel2)
                model = KerasClassifier(build_fn=dfpl.defineNNmodel)

                parameters = {'batch_size': batchSizes}

                clf = GridSearchCV(model, parameters, verbose=0)
                clf.fit(X_train, y_train)

                # save best estimator for epoochs/batchSize tuning per target
                modelfilepathW = args.p[0] + '/model.tuning-BatchSize.' + target + '.weights.h5'
                modelfilepathM = args.p[0] + '/model.tuning-BatchSize.' + target + '.json'
                outfilepath = args.p[0] + re.sub(".csv", '.hpTuningResults.01-BatchSize' + target + '.csv',
                                                 os.path.basename(args.i[0]))
                # outfilepath = '/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.ER.txt'
                clf.best_estimator_.model.save(filepath=modelfilepathM)
                clf.best_estimator_.model.save_weights(filepath=modelfilepathW)

                df = pd.concat([pd.DataFrame(clf.cv_results_['param_batch_size']),
                                pd.DataFrame(clf.cv_results_['mean_test_score']),
                                pd.DataFrame(clf.cv_results_['std_test_score']),
                                pd.DataFrame(clf.cv_results_['rank_test_score'])],
                               axis=1)
                df.columns = ['epoch', 'batch_size', 'mean_test_score', 'std_test_score', 'rank_test_score']
                df.to_csv(outfilepath, header=True)

                selected_bs = clf.best_params_['batch_size']
                selected_epochs = clf.best_params_['epochs']

            else:
                selected_bs = batchSizes[0]
                selected_epochs = epochs[0]

            # Hypertune optimizers
            optimizers = ['SGD', 'Adam']
            #optimizers = args.optimizers

            if optimizers.__len__() > 1:
                model = KerasClassifier(build_fn=dfpl.defineNNmodel, epochs=selected_epochs, batch_size=selected_bs)
                parameters = {'optimizer': optimizers} #['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']}
                clf = GridSearchCV(model, parameters, verbose=0)
                clf.fit(X_train, y_train)

                # save best estimator for optimizer tuning per target
                modelfilepathW = args.p[0] + '/model.tuning-optimizer.' + target + '.weights.h5'
                modelfilepathM = args.p[0] + '/model.tuning-optimizer.' + target + '.json'
                outfilepath = args.p[0] + re.sub(".csv", '.hpTuningResults.02-Optimizer' + target + '.csv',
                                                 os.path.basename(args.i[0]))

                clf.best_estimator_.model.save(filepath=modelfilepathM)
                clf.best_estimator_.model.save_weights(filepath=modelfilepathW)

                df = pd.concat([pd.DataFrame(clf.cv_results_['param_optimizer']),
                                pd.DataFrame(clf.cv_results_['mean_test_score']),
                                pd.DataFrame(clf.cv_results_['std_test_score']),
                                pd.DataFrame(clf.cv_results_['rank_test_score'])],
                               axis=1)
                df.columns = ['optimizer', 'mean_test_score', 'std_test_score', 'rank_test_score']
                df.to_csv(outfilepath, header=True)

                selected_optimizer = clf.best_params_['optimizer']

            else:
                selected_optimizer = optimizers[0]

            # Hypertune activation functions
            activations = args.activations #'sigmoid', 'tanh', 'relu'

            if activations.__len__() > 1:
                model = KerasClassifier(build_fn=dfpl.defineNNmodel, epochs=selected_epochs, batch_size=selected_bs)
                parameters = {'optimizer': [selected_optimizer],
                              'activation': args.activations}
                clf = GridSearchCV(model, parameters, verbose=0)
                clf.fit(X_train, y_train)

                # save best estimator for optimizer tuning per target
                modelfilepathW = args.p[0] + '/model.tuning-activation.' + target + '.weights.h5'
                modelfilepathM = args.p[0] + '/model.tuning-activation.' + target + '.json'
                outfilepath = args.p[0] + re.sub(".csv", '.hpTuningResults.03-Activation' + target + '.csv',
                                                 os.path.basename(args.i[0]))

                clf.best_estimator_.model.save(filepath=modelfilepathM)
                clf.best_estimator_.model.save_weights(filepath=modelfilepathW)

                df = pd.concat([pd.DataFrame(clf.cv_results_['param_optimizer']),
                                pd.DataFrame(clf.cv_results_['param_activation']),
                                pd.DataFrame(clf.cv_results_['mean_test_score']),
                                pd.DataFrame(clf.cv_results_['std_test_score']),
                                pd.DataFrame(clf.cv_results_['rank_test_score'])],
                               axis=1)
                df.columns = ['optimizer', 'activation', 'mean_test_score', 'std_test_score', 'rank_test_score']
                df.to_csv(outfilepath, header=True)

                selected_activation = clf.best_params_['activation']

            else:
                selected_activation = activations[0]

            # Maybe also optimize weight initializations??
            #inits = ['glorot_uniform', 'normal', 'uniform']


            ### find best performing parameters
            sys.stdout.write("Calculation time: %s min\n\n" % str(round((time()-start)/60, ndigits=2)))

        else:
            sys.stderr.write("ERROR: the target that you provide (%s) "
                  "is not contained in your data file (%s)" %
                  (target, args.i[0]))

