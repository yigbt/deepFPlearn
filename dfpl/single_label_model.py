import math
import logging
import shutil
import sys
from os import path
from time import time

import numpy as np
import pandas as pd
from keras import metrics
import tensorflow_addons as tfa
from tensorflow.keras import optimizers
from keras import regularizers
from keras.layers import Dense, Dropout, AlphaDropout
from keras.models import Model
# for NN model functions
from keras.models import Sequential
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow import keras

from dfpl import callbacks as cb
from dfpl import options
from dfpl import plot as pl
from dfpl import settings


def prepare_nn_training_data(df: pd.DataFrame, target: str, opts: options.Options) -> (np.ndarray,
                                                                                       np.ndarray,
                                                                                       options.Options):
    # check the value counts and abort if too imbalanced
    allowed_imbalance = 0.1

    if opts.compressFeatures:
        vc = df[df[target].notna() & df["fpcompressed"].notnull()][target].value_counts()
    else:
        vc = df[df[target].notna() & df["fp"].notnull()][target].value_counts()
    if min(vc) < max(vc) * allowed_imbalance:
        logging.info(
            f" Your training data is extremely unbalanced ({target}): 0 - {vc[0]}, and 1 - {vc[1]} values.")
        if opts.sampleDown:
            logging.info(f" I will downsample your data")
            opts.sampleFractionOnes = allowed_imbalance
        else:
            logging.info(f" I will not downsample your data automatically.")
            logging.info(f" Consider to enable down sampling of the 0 values with --sampleDown option.")

    logging.info("Preparing training data matrices")
    if opts.compressFeatures:
        logging.info("Using compressed fingerprints")
        df_fpc = df[df[target].notna() & df["fpcompressed"].notnull()]
        logging.info(f"DataSet has {str(df_fpc.shape)} entries with not NA values in fpcompressed and {target}")
        if opts.sampleDown:
            assert 0.0 < opts.sampleFractionOnes < 1.0
            logging.info(f"Using fractional sampling {opts.sampleFractionOnes}")
            # how many ones
            counts = df_fpc[target].value_counts()
            logging.info(f"Number of fraction sampling values: {str(counts)}")

            # add sample of 0s to df of 1s
            dfX = df_fpc[df_fpc[target] == 1].append(
                df_fpc[df_fpc[target] == 0].sample(
                    int(min(counts[0], counts[1] / opts.sampleFractionOnes))
                )
            )
            x = np.array(
                dfX["fpcompressed"].to_list(),
                dtype=settings.nn_fp_compressed_numpy_type,
                copy=settings.numpy_copy_values
            )
            y = np.array(
                dfX[target].to_list(),
                dtype=settings.nn_target_numpy_type,
                copy=settings.numpy_copy_values
            )
        else:
            logging.info("Fraction sampling is OFF")
            # how many ones, how many zeros
            counts = df_fpc[target].value_counts()
            logging.info(f"Number of values (total): 0 - {str(counts[0])}, 1 - {str(counts[1])}")

            x = np.array(
                df_fpc["fpcompressed"].to_list(),
                dtype=settings.nn_fp_compressed_numpy_type,
                copy=settings.numpy_copy_values
            )
            y = np.array(
                df_fpc[target].to_list(),
                dtype=settings.nn_target_numpy_type,
                copy=settings.numpy_copy_values
            )
    else:
        logging.info("Using uncompressed fingerprints")
        df_fp = df[df[target].notna() & df["fp"].notnull()]
        logging.info(f"DataSet has {str(df_fp.shape)} entries with not NA values in fpcompressed and {target}")
        if opts.sampleFractionOnes:
            logging.info(f"Using fractional sampling {opts.sampleFractionOnes}")
            counts = df_fp[target].value_counts()
            logging.info(f"Number of fraction sampling values: {str(counts)}")

            dfX = df_fp[df_fp[target] == 1.0].append(
                df_fp[df_fp[target] == 0.0].sample(
                    int(min(counts[0], counts[1] / opts.sampleFractionOnes)))
            )
            x = np.array(dfX["fp"].to_list(), dtype=settings.ac_fp_numpy_type, copy=settings.numpy_copy_values)
            y = np.array(dfX[target].to_list(), copy=settings.numpy_copy_values)
        else:
            logging.info("Fraction sampling is OFF")
            # how many ones, how many zeros
            counts = df_fp[target].value_counts()
            logging.info(f"Number of values (total): 0 - {str(counts[0])}, 1 - {str(counts[1])}")

            x = np.array(df_fp["fp"].to_list(), dtype=settings.ac_fp_numpy_type, copy=settings.numpy_copy_values)
            y = np.array(df_fp[target].to_list(), copy=settings.numpy_copy_values)
    return x, y, opts


def define_single_label_model(input_size: int,
                              opts: options.Options) -> Model:
    lf = dict({"mse": "mean_squared_error",
               "bce": "binary_crossentropy"})

    if opts.optimizer == 'Adam':
        my_optimizer = optimizers.Adam(learning_rate=opts.learningRate)
    elif opts.optimizer == 'SGD':
        my_optimizer = optimizers.SGD(lr=opts.learningRate, momentum=0.9)
    else:
        logging.error(f"Your selected optimizer is not supported:{opts.optimizer}.")
        sys.exit("Unsupported optimizer.")

    my_hidden_layers = {"2048": 6, "1024": 5, "999": 5, "512": 4, "256": 3, "16": 3}

    if not str(input_size) in my_hidden_layers.keys():
        raise ValueError("Wrong input-size. Must be in {2048, 1024, 999, 512, 256}.")

    nhl = int(math.log2(input_size) / 2 - 1)

    model = Sequential()
    # From input to 1st hidden layer
    if opts.activationFunction == "relu":
        model.add(Dense(units=int(input_size / 2),
                        input_dim=input_size,
                        activation="relu",
                        kernel_regularizer=regularizers.l2(opts.l2reg),
                        kernel_initializer="he_uniform"))
        model.add(Dropout(opts.dropout))
    if opts.activationFunction == "selu":
        model.add(Dense(units=int(input_size / 2),
                        input_dim=input_size,
                        activation="selu",
                        kernel_initializer="lecun_normal"))
        model.add(AlphaDropout(opts.dropout))
    else:
        logging.error("Only 'relu' and 'selu' activation is supported")
        sys.exit(-1)

    # next hidden layers
    for i in range(1, nhl):
        factor_units = 2 ** (i + 1)
        factor_dropout = 2 * i
        if opts.activationFunction == "relu":
            model.add(Dense(units=int(input_size / factor_units),
                            activation="relu",
                            kernel_regularizer=regularizers.l2(opts.l2reg),
                            kernel_initializer="he_uniform"))
            model.add(Dropout(opts.dropout / factor_dropout))
        if opts.activationFunction == "selu":
            model.add(Dense(units=int(input_size / factor_units),
                            activation="selu",
                            kernel_initializer="lecun_normal"))
            model.add(AlphaDropout(opts.dropout / factor_dropout))
        else:
            logging.error("Only 'relu' and 'selu' activation is supported")
            sys.exit(-1)

    # output layer
    model.add(Dense(units=1,
                    activation='sigmoid'))

    model.summary(print_fn=logging.info)

    # compile model
    model.compile(loss=lf[opts.lossFunction],
                  optimizer=my_optimizer,
                  metrics=['accuracy',
                           tfa.metrics.F1Score(num_classes=1, threshold=0.5, average="weighted"),
                           metrics.Precision(),
                           metrics.Recall()]
                  )
    return model


def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, file_prefix: str, model: Model,
                   target: str, fold: int) -> pd.DataFrame:
    name = path.basename(file_prefix).replace("_", " ")
    performance = pd.DataFrame()

    # predict test set to compute MCC, AUC, ROC curve, etc. (see below)
    y_predict = model.predict(X_test).flatten()
    y_predict_int = [int(round(y)) for y in y_predict]
    y_test_int = [int(y) for y in y_test]
    pd.DataFrame({"y_true": y_test, "y_predicted": y_predict,
                  "y_true_int": y_test_int, "y_predicted_int": y_predict_int,
                  "target": target, "fold": fold}). \
        to_csv(path_or_buf=f"{file_prefix}.predicted.testdata.csv")

    # compute the metrics that depend on the class label
    precicion_recall = classification_report(y_test_int, y_predict_int, output_dict=True)
    prf = pd.DataFrame.from_dict(precicion_recall)[['0', '1']]
    prf.to_csv(path_or_buf=f"{file_prefix}.predicted.testdata.prec_rec_f1.csv")

    # validation loss, accuracy
    loss, acc = tuple(model.evaluate(x=X_test, y=y_test)[:2])
    # MCC
    MCC = matthews_corrcoef(y_test_int, y_predict_int)
    # generate the AUC-ROC curve data from the validation data
    FPR, TPR, thresholds_keras = roc_curve(y_true=y_test_int,
                                           y_score=y_predict_int,
                                           drop_intermediate=False)
    AUC = auc(FPR, TPR)
    # save AUC data to csv
    pd.DataFrame(list(zip(FPR, TPR, [AUC] * len(FPR), [target] * len(FPR), [fold] * len(FPR))),
                 columns=['fpr', 'tpr', 'auc_value', 'target', 'fold']). \
        to_csv(path_or_buf=f"{file_prefix}.predicted.testdata.aucdata.csv")

    pl.plot_auc(fpr=FPR, tpr=TPR, target=target, auc_value=AUC,
                filename=f"{file_prefix}.predicted.testdata.aucdata.svg")

    # [[tn, fp]
    #  [fn, tp]]
    cfm = confusion_matrix(y_true=y_test_int,
                           y_pred=y_predict_int)

    logging.info(f"Model '{name}' (trained) evaluation on test data")
    logging.info(f"[loss, accuracy, precision, recall]: "
                 f"{[round(i, ndigits=4) for i in performance]}")
    logging.info('MCC: ' + str(MCC.__round__(2)))
    logging.info('CFM: \n\tTN=' + str(cfm[0][0]) + '\tFP=' + str(cfm[0][1]) + '\n\tFN=' + str(cfm[1][0]) +
                 '\tTP=' + str(cfm[1][1]))

    return pd.DataFrame.from_dict({'p_0': prf['0']['precision'],
                                   'r_0': prf['0']['recall'],
                                   'f1_0': prf['0']['f1-score'],
                                   'p_1': prf['1']['precision'],
                                   'r_1': prf['1']['recall'],
                                   'f1_1': prf['1']['f1-score'],
                                   'loss': loss,
                                   'accuracy': acc,
                                   'MCC': MCC,
                                   'AUC': AUC,
                                   'target': target,
                                   'fold': fold}, orient='index').T


def fit_and_evaluate_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                           fold: int, target: str, opts: options.Options) -> pd.DataFrame:
    logging.info("Training of fold number:" + str(fold))
    logging.info(f"The distribution of 0 and 1 values is:")
    logging.info(f"\ttrain data:\t{pd.DataFrame(y_train)[0].value_counts().to_list()}")
    logging.info(f"\ttest  data:\t{pd.DataFrame(y_test)[0].value_counts().to_list()}")

    model_file_prefix = path.join(opts.outputDir, f"{target}_single-labeled_Fold-{fold}")

    model = define_single_label_model(input_size=X_train.shape[1],
                                      opts=opts)

    callback_list = cb.nn_callback(checkpoint_path=f"{model_file_prefix}.model.weights.hdf5",
                                   opts=opts)

    # measure the training time
    start = time()
    hist = model.fit(X_train, y_train,
                     callbacks=callback_list,
                     epochs=opts.epochs,
                     batch_size=opts.batchSize,
                     verbose=opts.verbose,
                     # validation_split=opts.testSize
                     validation_data=(X_test, y_test)
                     )
    trainTime = str(round((time() - start) / 60, ndigits=2))
    logging.info(f"Computation time for training the single-label model for {target}: {trainTime} min")

    # save and plot history
    pd.DataFrame(hist.history).to_csv(path_or_buf=f"{model_file_prefix}.history.csv")
    pl.plot_history(history=hist, file=f"{model_file_prefix}.history.svg")
    # evaluate model
    performance = evaluate_model(X_test=X_test, y_test=y_test, file_prefix=model_file_prefix, model=model,
                                 target=target, fold=fold)

    return performance


def train_single_label_models(df: pd.DataFrame, opts: options.Options) -> None:
    """
    Train individual models for all targets (columns) present in the provided target data (y) and a multi-label
    model that classifies all targets at once. For each individual target the data is first subset to exclude NA
    values (for target associations). A random sample of the remaining data (size is the split fraction) is used for
    training and the remaining data for validation.

    :param opts: The command line arguments in the options class
    :param df: The dataframe containing x matrix and at least one column for a y target.
    """

    # find target columns
    names_y = [c for c in df.columns if c not in ['cid', 'id', 'mol_id', 'smiles', 'fp', 'inchi', 'fpcompressed']]

    if opts.wabTracking:
        # For W&B tracking, we only train one target.
        # We love to have the ER target, but in case it's not there, we use the first one available
        if opts.wabTarget in names_y:
            names_y = [opts.wabTarget]
        else:
            logging.error(f"The specified wabTarget for Weights & Biases tracking does not exist: {opts.wabTarget}")
            names_y = [names_y[0]]

    model_evaluation = pd.DataFrame(
        columns=["p_0", "r_0", "f1_0", "p_1", "r_1", "f1_1",  # class label dependent metrics
                 "loss", "accuracy", "MCC", "AUC", "target", "fold"])  # total metrics

    # For each individual target train a model
    for target in names_y:  # [:1]:
        # target=names_y[1] # --> only for testing the code
        x, y, opts = prepare_nn_training_data(df, target, opts)
        if x is None:
            continue

        logging.info(f"X training matrix of shape {x.shape} and type {x.dtype}")
        logging.info(f"Y training matrix of shape {y.shape} and type {y.dtype}")

        if opts.kFolds == 1:
            # train a single 'fold'
            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=opts.testSize)
            performance = fit_and_evaluate_model(X_train=xtrain, X_test=xtest, y_train=ytrain, y_test=ytest,
                                                 fold=0, target=target, opts=opts)
            model_evaluation = model_evaluation.append(performance, ignore_index=True)
            # save complete model
            trained_model = define_single_label_model(input_size=x[0].__len__(),
                                                      opts=opts)
            trained_model.load_weights(path.join(opts.outputDir,
                                                 f"{target}_single-labeled_Fold-0.model.weights.hdf5"))
            # create output directory and store complete model
            trained_model.save(filepath=path.join(opts.outputDir,
                                                  f"{target}_saved_model"))
        elif 1 < opts.kFolds < int(x.shape[0] / 100):
            # do a kfold cross-validation
            kfold_c_validator = KFold(n_splits=opts.kFolds, shuffle=True, random_state=42)
            fold_no = 1
            # split the data
            for train, test in kfold_c_validator.split(x, y):
                # for testing use one of the splits:
                # kf = kfold_c_validator.split(x, y)
                # train, test = next(kf)
                performance = fit_and_evaluate_model(X_train=x[train], X_test=x[test],
                                                     y_train=y[train], y_test=y[test],
                                                     fold=fold_no, target=target, opts=opts)
                model_evaluation = model_evaluation.append(performance, ignore_index=True)
                fold_no += 1
                # now next fold

            # select and copy best model - how to define the best model?
            best_fold = model_evaluation.sort_values(by=['p_1', 'r_1', 'MCC'],
                                                     ascending=False,
                                                     ignore_index=True)['fold'][0]
            # copy checkpoint model weights
            shutil.copy(
                src=path.join(opts.outputDir, f"{target}_single-labeled_Fold-{best_fold}.model.weights.hdf5"),
                dst=path.join(opts.outputDir, f"{target}_single-labeled_Fold-{best_fold}.best.model.weights.hdf5"))
            # save complete model
            best_model = define_single_label_model(input_size=x[0].__len__(),
                                                   opts=opts)
            best_model.load_weights(path.join(opts.outputDir,
                                              f"{target}_single-labeled_Fold-{best_fold}.model.weights.hdf5"))
            # create output directory and store complete model
            best_model.save(filepath=path.join(opts.outputDir,
                                               f"{target}_saved_model"))
        else:
            logging.info("Your selected number of folds for Cross validation is out of range. "
                         "It must be 1 or smaller than 1 hundredth of the number of samples.")
            exit(1)

        # now next target
    # store the evaluation data of all trained models (all targets, all folds)
    model_evaluation.to_csv(path_or_buf=path.join(opts.outputDir, 'single_label_model.evaluation.csv'))
