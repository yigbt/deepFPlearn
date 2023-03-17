from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def tuning_model(optimizer, activation, init, dropout=0.2):
    """
    model for tuning optimizer, activation functions and initialization of hidden layers

    :param optimizer:
    :param activation:
    :param init:
    :param dropout:
    :return:
    """
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

# this stores the best performing parameters for each model for each target
results = pd.DataFrame()

np.random.seed(0)

modelfilepathprefix = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/"

filepath = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
outfilepath = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.HPtuning.results.txt"
dataset = pd.read_csv(filepath)

le = LabelEncoder()

# which target column to use? put this in a loop later for all targets
# here we need a for loop
for target in dataset.columns[1:7]:
    # target = dataset.columns[1] # 'AR'

    modelfilepathW = str(modelfilepathprefix) + '/model.' + target + '.weights.h5'
    modelfilepathM = str(modelfilepathprefix) + '/model.' + target + '.json'

    tmp = dataset[target].astype('category')
    Y = np.asarray(tmp)
    naRows = np.isnan(Y)

    d = dataset[~naRows]

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

    model = KerasClassifier(build_fn=tuning_model, batch_size=20)

    batch_sizes = [20]
    epochs = [30, 100]
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

    # find best performing parameters
    file = open(outfilepath, "a")
    file.write("# --------------------------------------------------------------------------- #\n")
    file.write("### Results for %s target ###\n" % target)
    file.write("Best: %f using %s\n" % (clf_results.best_score_, clf_results.best_params_))
    file.write("Calculation time: %s min\n\n" % str(round((time() - start) / 60, ndigits=2)))
    file.close()
