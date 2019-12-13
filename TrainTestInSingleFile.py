import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from time import time

# --------------------------------------------------------------------------- #

# initial model, and model for tuning batchsize and epochs
def c_model():
    model = Sequential()
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(341, activation='relu'))
    model.add(Dense(85, activation='relu'))
    model.add(Dense(17, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --------------------------------------------------------------------------- #

# model for tuning optmizer and activation functions of hidden layers
def tuning_model(optimizer, activationf, init, dropout=0.2):
    model = Sequential()
    model.add(Dense(1024, activation=activationf, init=init))
    model.add(Dropout(dropout))
    model.add(Dense(341, activation=activationf, init=init))
    model.add(Dropout(dropout))
    model.add(Dense(85, activation=activationf, init=init))
    model.add(Dropout(dropout))
    model.add(Dense(17, activation=activationf, init=init))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# --------------------------------------------------------------------------- #

# this stores the best performing parameters for each model for each target
results = pd.DataFrame()

np.random.seed(0)

filepath = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
outfilepath = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.HPtuning.results.csv"
dataset = pd.read_csv(filepath)

le = LabelEncoder()

# which target column to use? put this in a loop later for all targets
# here we need a for loop
for target in dataset.columns:
    #target = dataset.columns[1] # 'AR'

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
        X[i] = list(map(int, [char for char in fp]))

    # generate y vector(s)
    y = to_categorical(np.array(d['outcome']), num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    start = time()


    ### find best performing parameters
    print(clf.best_score_, clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    parameters = clf.cv_results_['params']
    for mean, parameter in zip(means, parameters):
        print(mean, parameters)

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
    clf = GridSearchCV(model, parameters)
    clf_results = clf.fit(X_train, y_train)

    ### find best performing parameters
    print("# --------------------------------------------------------------------------- #")
    print("### Results for %s target ###" % target)
    print("Best: %f using %s" % (clf_results.best_score_, clf_results.best_params_))
    print("Calculation time: %s min" % str(round((time()-start)/60, ndigits=2)))



