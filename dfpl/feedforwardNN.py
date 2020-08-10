import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import shutil

# for NN model functions
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras import optimizers
from keras.optimizers import SGD

from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc

# from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping

import dfpl.options
import dfpl.autoencoder as ac

from time import time


def defineNNmodel(
        input_size: int = 2048,
        l2reg: float = 0.001,
        dropout: float = 0.2,
        activation: str = 'relu',
        optimizer: str = 'Adam',
        lr: float = 0.001,
        decay: float = 0.01) -> Model:
    """

    :param input_size:
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

    if not str(input_size) in myhiddenlayers.keys():
        raise ValueError("Wrong input-size. Must be in {2048, 1024, 999, 512, 256}.")

    nhl = int(math.log2(input_size) / 2 - 1)

    model = Sequential()
    # From input to 1st hidden layer
    model.add(Dense(units=int(input_size / 2), input_dim=input_size,
                    activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    # next hidden layers
    for i in range(1, nhl):
        factorunits = 2 ** (i + 1)
        factordropout = 2 * i
        model.add(Dense(units=int(input_size / factorunits),
                        activation=activation,
                        kernel_regularizer=regularizers.l2(l2reg)))
        model.add(Dropout(dropout / factordropout))
    # output layer
    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()

    # compile model
    model.compile(loss="mse", optimizer=myoptimizer, metrics=['accuracy'])

    return model


def defineOutfileNames(pathprefix: str, target: str, fold: int) -> tuple:
    """
    This function returns the required paths for output files or directories.

    :param pathprefix: A file path prefix for all files.
    :param target: The name of the target.
    :param fold:

    :return: A tuple of 14 output file names.
    """

    modelname = "/" + target + '.Fold-' + str(fold)

    modelfilepathW = str(pathprefix) + modelname + '.weights.h5'
    modelfilepathM = str(pathprefix) + modelname + '.json'
    modelhistplotpathL = str(pathprefix) + modelname + '.loss.svg'
    modelhistplotpathA = str(pathprefix) + modelname + '.acc.svg'
    modelhistplotpath = str(pathprefix) + modelname + '.history.svg'
    modelhistcsvpath = str(pathprefix) + modelname + '.history.csv'
    modelvalidation = str(pathprefix) + modelname + '.validation.csv'
    modelAUCfile = str(pathprefix) + modelname + '.auc.svg'
    modelAUCfiledata = str(pathprefix) + modelname + '.auc.data.csv'
    outfilepath = str(pathprefix) + modelname + '.trainingResults.txt'
    checkpointpath = str(pathprefix) + modelname + '.checkpoint.model.hdf5'
    modelheatmapX = str(pathprefix) + modelname + '.heatmap.X.svg'
    modelheatmapZ = str(pathprefix) + modelname + '.AC.heatmap.Z.svg'

    return (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
            modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
            modelAUCfiledata, outfilepath, checkpointpath,
            modelheatmapX, modelheatmapZ)


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
    trainedmodel = defineNNmodel(input_size=Z_test.shape[1])

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
    MCC = matthews_corrcoef(ytrueInt, predictionsInt)

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


def trainNNmodels(df: pd.DataFrame,
                  opts: dfpl.options.TrainOptions,
                  usecompressed: bool) -> None:
    #
    # modelfilepathprefix: str, x: pd.DataFrame, y: pd.DataFrame,
    # split: float = 0.2, epochs: int = 50, params: str = None,
    # verbose: int = 2, kfold: int = 5) -> None:
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
    :param verbose: Verbosity level.

    :return: A list with loss and accuracy values for each individual model.
    """

    # find target columns
    namesY = [c for c in df.columns if c not in ['id', 'smiles', 'fp', 'inchi', 'fpcompressed']]


    ### For each individual target (+ summarized target)
    # for target in y.columns:  # [:1]:
    for target in namesY:#[:2]:
        # target=namesY[0] # --> only for testing the code

        if usecompressed:
            x = np.array(df[df[target].notna() & df['fpcompressed'].notnull()]["fpcompressed"].to_list())
        else:
            x = np.array(df[df[target].notna() & df['fp'].notnull()]["fp"].to_list())

        y = np.array(df[df[target].notna() & df['fp'].notnull()][target].to_list())

        # do a kfold cross validation for the FNN training
        kfoldCValidator = KFold(n_splits=opts.kFolds,
                                shuffle=True,
                                random_state=42)

        # store acc and loss for each fold
        allscores = pd.DataFrame(columns=["fold_no",  # fold number of k-fold CV
                                          "loss", "val_loss", "acc", "val_acc",  # FNN training
                                          "loss_test", "acc_test", "mcc_test"])  # FNN test data

        fold_no = 1

        # split the data
        for train, test in kfoldCValidator.split(x, y):  # kfoldCValidator.split(Xt, Yt):
            # for testing use one of the splits:
            # kf = kfoldCValidator.split(x, y)
            # train, test = next(kf)

            if opts.verbose > 0:
                logging.info("Training of fold number:" + str(fold_no))

            # define all the output file/path names
            (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
             modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
             modelAUCfiledata, outfilepath, checkpointpath,
             modelheatmapX, modelheatmapZ) = defineOutfileNames(pathprefix=opts.outputDir,
                                                                target=target + "_compressed-" + str(usecompressed),
                                                                fold=fold_no)

            model = defineNNmodel(input_size=x[train].shape[1])

            callback_list = ac.autoencoderCallback(checkpointpath=checkpointpath, patience=20)
            # callback_list = defineCallbacks(checkpointpath=checkpointpath, patience=20,
            #                                 rlrop=True, rlropfactor=0.1, rlroppatience=100)
            # measure the training time
            start = time()
            # train and validate
            hist = model.fit(x[train], y[train],
                             callbacks=callback_list,
                             epochs=opts.epochs, batch_size=256, verbose=opts.verbose,
                             validation_split=opts.testingFraction)
            #                             validation_data=(Z_test, y_test))  # this overwrites val_split!
            trainTime = str(round((time() - start) / 60, ndigits=2))

            if opts.verbose > 0:
                logging.info("Computation time for training the single-label FNN:" + trainTime + "min")

            # validate model on test data set (x_test, y_test)
            scores = validateModelOnTestData(x[test], checkpointpath, y[test],
                                             "FNN", modelvalidation, target,
                                             modelAUCfiledata, modelAUCfile)

            idx = hist.history['val_loss'].index(min(hist.history['val_loss']))

            row_df = pd.DataFrame([[fold_no,
                                    hist.history['loss'][idx], hist.history['val_loss'][idx],
                                    hist.history['accuracy'][idx], hist.history['val_accuracy'][idx],
                                    scores[0], scores[1], scores[2]]],
                                  columns=["fold_no",  # fold number of k-fold CV
                                           "loss", "val_loss", "acc", "val_acc",  # FNN training
                                           "loss_test", "acc_test", "mcc_test"]
                                  )
            print(row_df)
            allscores = allscores.append(row_df, ignore_index=True)
            fold_no += 1
            del model
            # now next fold

        print(allscores)

        # finalize model
        # 1. provide best performing fold variant
        # select best model based on MCC
        idx2 = allscores[['mcc_test']].idxmax().ravel()[0]
        fold_no = allscores._get_value(idx2, 'fold_no')

        modelname = target + "_compressed-" + str(usecompressed) + '.Fold-' + str(fold_no)
        checkpointpath = str(opts.outputDir) + "/" + modelname + '.checkpoint.model.hdf5'

        bestModelfile = checkpointpath.replace("Fold-" + str(fold_no) + ".checkpoint.", "best.FNN-")

        # store all scores
        file = re.sub("\.hdf5", "scores.csv", re.sub("Fold-.\.checkpoint", "Fold-All", checkpointpath))
        allscores.to_csv(file)

        # copy best DNN model
        shutil.copyfile(checkpointpath, bestModelfile)
        logging.info("Best model for FNN is saved: " + bestModelfile)

        # AND retrain with full data set
        fullModelfile = checkpointpath.replace("Fold-" + str(fold_no) + ".checkpoint", "full.FNN-")
        # measure the training time
        start = time()

        model = defineNNmodel(input_size=x.shape[1])  # X_train.shape[1])
        callback_list = ac.autoencoderCallback(checkpointpath=fullModelfile, patience=20)

        # train and validate
        hist = model.fit(x, y,
                         callbacks=callback_list,
                         epochs=opts.epochs, batch_size=256, verbose=opts.verbose,
                         validation_split=opts.testingFraction)
        #                             validation_data=(Z_test, y_test))  # this overwrites val_split!
        trainTime = str(round((time() - start) / 60, ndigits=2))

        if opts.verbose > 0:
            logging.info("Computation time for training the full classification FNN: " + trainTime + "min")
        # plotHistoryVis(hist,
        #                modelhistplotpath.replace("Fold-" + str(fold_no), "full.DNN-model"),
        #                modelhistcsvpath.replace("Fold-" + str(fold_no), "full.DNN-model"),
        #                modelhistplotpathA.replace("Fold-" + str(fold_no), "full.DNN-model"),
        #                modelhistplotpathL.replace("Fold-" + str(fold_no), "full.DNN-model"), target)
        # print(f'[INFO]: Full model for DNN is saved:\n        - {fullModelfile}')

        pd.DataFrame(hist.history).to_csv(fullModelfile.replace(".hdf5", ".history.csv"))

        del model
        # now next target


def defineNNmodelMulti(inputSize=2048,
                       outputSize=None,
                       l2reg=0.001,
                       dropout=0.2,
                       activation='relu',
                       optimizer='Adam',
                       lr=0.001,
                       decay=0.01):
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


def validateMultiModelOnTestData(Z_test, checkpointpath, y_test, colnames, resultfile):
    # load checkpoint model with min(val_loss)
    trainedmodel = defineNNmodelMulti(inputSize=Z_test.shape[1], outputSize=y_test.shape[1])

    # predict values with random model
    predictions_random = pd.DataFrame(trainedmodel.predict(Z_test), columns=[n + '-predRandom' for n in colnames])
    # predictions_random = pd.DataFrame(trainedmodel.predict(Z_test), columns=colnames + '-predRandom')

    # load weights into random model
    trainedmodel.load_weights(checkpointpath)

    # predict with trained model
    predictions = pd.DataFrame(trainedmodel.predict(Z_test),
                               columns=[n + '-pred' for n in colnames])
    scores = pd.DataFrame((predictions.round() == y_test).sum() / y_test.shape[0], columns=['correctPredictions'])

    results = pd.concat([predictions_random, predictions, pd.DataFrame(y_test,
                                                                       columns=[n + '-true' for n in colnames])],
                        axis=1)
    results.to_csv(resultfile)

    return scores


def trainNNmodelsMulti(df: pd.DataFrame,
                       opts: dfpl.options.TrainOptions,
                       usecompressed: bool) -> None:
    # modelfilepathprefix: str, x: pd.DataFrame, y: pd.DataFrame,
    # split: float = 0.2, epochs: int = 500,
    # verbose: int= 2, kfold: int = 5) -> None:

    # find target columns
    namesY = [c for c in df.columns if c not in ['id', 'smiles', 'fp', 'inchi']]
    idxnotNA = df[namesY].dropna().index

    if usecompressed:
        # get compressed fingerprints as numpy array
        fpMatrix = np.array(df.iloc[idxnotNA][df['fpcompressed'].notnull()]['fpcompressed'].to_list())
        y = np.array(df.iloc[idxnotNA][df['fpcompressed'].notnull()][namesY])
    else:
        # get fingerprints as numpy array
        fpMatrix = np.array(df.iloc[idxnotNA][df['fp'].notnull()]['fp'].to_list())
        y = np.array(df.iloc[idxnotNA][df['fp'].notnull()][namesY])

    # do a kfold cross validation for the autoencoder training
    kfoldCValidator = KFold(n_splits=opts.kFolds,
                            shuffle=True,
                            random_state=42)

    # store acc and loss for each fold
    allscores = pd.DataFrame(columns=["fold_no",  # fold number of k-fold CV
                                      "loss", "val_loss", "acc", "val_acc"]) #,  # FNN training
                                      # "loss_test", "acc_test", "mcc_test"])  # FNN test data

    fold_no = 1

    # split the data
    for train, test in kfoldCValidator.split(fpMatrix, y):
        #kf = kfoldCValidator.split(fpMatrix, y)
        #train, test = next(kf)
        # define all the output file/path names
        (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
         modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
         modelAUCfiledata, outfilepath, checkpointpath,
         modelheatmapX, modelheatmapZ) = defineOutfileNames(pathprefix=opts.outputDir,
                                                            target="multi" + "_compressed-" + str(usecompressed),
                                                            fold=fold_no)

        # use a dnn for multi-class prediction
        model = defineNNmodelMulti(inputSize=fpMatrix[train].shape[1],
                                   outputSize=y.shape[1])

        callback_list = ac.autoencoderCallback(checkpointpath=checkpointpath,
                                               patience=20)
        # measure the training time
        start = time()

        # train and validate
        hist = model.fit(fpMatrix[train], y[train],
                         callbacks=callback_list,
                         epochs=opts.epochs,
                         batch_size=256,
                         verbose=opts.verbose,
                         validation_split=opts.testingFraction)

        trainTime = str(round((time() - start) / 60, ndigits=2))

        if opts.verbose > 0:
            logging.info("Computation time for training the multi-label FNN: " + trainTime + " min")

        # validate model on test data set (x_test, y_test)
        scores = validateMultiModelOnTestData(Z_test=fpMatrix[test],
                                              checkpointpath=checkpointpath,
                                              y_test=y[test],
                                              colnames=namesY,
                                              resultfile=outfilepath.replace("trainingResults.txt",
                                                                             "predictionResults.csv"))


        idx = hist.history['val_loss'].index(min(hist.history['val_loss']))
        row_df = pd.DataFrame([[fold_no,
                                hist.history['loss'][idx], hist.history['val_loss'][idx],
                                hist.history['accuracy'][idx], hist.history['val_accuracy'][idx]]],
                              columns=["fold_no",  # fold number of k-fold CV
                                       "loss", "val_loss", "acc", "val_acc"]
                              )#.join(scores.T)


        print(row_df)
        allscores = allscores.append(row_df, ignore_index=True)

        fold_no = fold_no + 1
        del model

    print(allscores)
#
# # finalize model
# # 1. provide best performing fold variant
# # select best model based on MCC
# idx2 = allscores[['mcc_test']].idxmax().ravel()[0]
# fold_no = allscores._get_value(idx2, 'fold_no')
#
# modelname = 'multi.Fold-' + str(fold_no)
# checkpointpath = str(modelfilepathprefix) + '.' + modelname + '.checkpoint.model.hdf5'
# bestModelfile = checkpointpath.replace("Fold-" + str(fold_no) + ".checkpoint.", "best.FNN-")
#
# file = re.sub("\.hdf5", "scores.csv", re.sub("Fold-.\.checkpoint", "Fold-All", checkpointpath))
# allscores.to_csv(file)
#
# # copy best DNN model
# shutil.copyfile(checkpointpath, bestModelfile)
# print(f'[INFO]: Best models for FNN is saved:\n        - {bestModelfile}')
#
# # AND retrain with full data set
# fullModelfile = checkpointpath.replace("Fold-" + str(fold_no) + ".checkpoint", "full.FNN-")
# # measure the training time
# start = time()
#
# model = defineNNmodel(inputSize=xmulti[train].shape[1])
# callback_list = defineCallbacks(checkpointpath=fullModelfile, patience=20,
#                                 rlrop=True, rlropfactor=0.1, rlroppatience=100)
# # train and validate
# hist = model.fit(xmulti, ymulti,
#                  callbacks=callback_list,
#                  epochs=epochs, batch_size=256, verbose=2, validation_split=split)
# #                             validation_data=(Z_test, y_test))  # this overwrites val_split!
# trainTime = str(round((time() - start) / 60, ndigits=2))
#
# if verbose > 0:
#     print(f"[INFO:] Computation time for training the full classification FNN: {trainTime} min")
# plotHistoryVis(hist,
#                modelhistplotpath.replace("Fold-" + str(fold_no), "full.DNN-model"),
#                modelhistcsvpath.replace("Fold-" + str(fold_no), "full.DNN-model"),
#                modelhistplotpathA.replace("Fold-" + str(fold_no), "full.DNN-model"),
#                modelhistplotpathL.replace("Fold-" + str(fold_no), "full.DNN-model"), target)
# print(f'[INFO]: Full models for DNN is saved:\n        - {fullModelfile}')
#
# pd.DataFrame(hist.history).to_csv(fullModelfile.replace(".hdf5", ".history.csv"))
# stats.append([target, [x.__round__(2) for x in scores]])
