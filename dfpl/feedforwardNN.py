import array
import logging
import math
import re
import shutil
import sys
from time import time
import os
from os import path
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
# for NN model functions
from tensorflow.keras.models import Sequential
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback


from dfpl import callbacks as cb
from dfpl import history as ht
from dfpl import options
from dfpl import settings


def define_out_file_names(path_prefix: str, target: str, fold: int = -1) -> tuple:
    """
    This function returns the required paths for output files or directories.

    :param path_prefix: A file path prefix for all files.
    :param target: The name of the target.
    :param fold:

    :return: A tuple of 14 output file names.
    """

    if fold == -1:
        model_name = "/" + target + '.Full'
    else:
        model_name = "/" + target + '.Fold-' + str(fold)

    model_file_path_weights = str(path_prefix) + model_name + '.weights.h5'
    model_file_path_json = str(path_prefix) + model_name + '.json'
    model_hist_path = str(path_prefix) + model_name
    model_hist_csv_path = str(path_prefix) + model_name + '.history.csv'
    model_predict_valset_csv_path = str(path_prefix) + model_name + '.predictValSet.csv'
    model_validation = str(path_prefix) + model_name + '.validation.csv'
    model_auc_file = str(path_prefix) + model_name + '.auc_value.svg'
    model_auc_file_data = str(path_prefix) + model_name + '.auc_value.data.csv'
    out_file_path = str(path_prefix) + model_name + '.trainingResults.txt'
    checkpoint_path = str(path_prefix) + model_name + '.checkpoint.model.hdf5'
    model_heatmap_x = str(path_prefix) + model_name + '.heatmap.X.svg'
    model_heatmap_z = str(path_prefix) + model_name + '.AC.heatmap.Z.svg'

    return (model_file_path_weights, model_file_path_json, model_hist_path,
            model_hist_csv_path, model_predict_valset_csv_path,
            model_validation, model_auc_file,
            model_auc_file_data, out_file_path, checkpoint_path,
            model_heatmap_x, model_heatmap_z)


def define_nn_multi_label_model(input_size: int,
                                output_size: int,
                                opts: options.Options) -> Model:
    if opts.optimizer == 'Adam':
        my_optimizer = optimizers.Adam(learning_rate=opts.learningRate)
    elif opts.optimizer == 'SGD':
        my_optimizer = optimizers.SGD(lr=opts.learningRate, momentum=0.9)
    else:
        logging.error(f"Your selected optimizer is not supported:{opts.optimizer}.")
        sys.exit("Unsupported optimizer.")

    nhl = int(math.log2(input_size) / 2 - 1)

    model = Sequential()
    # From input to 1st hidden layer
    model.add(Dense(units=int(input_size / 2),
                    input_dim=input_size,
                    activation=opts.activationFunction,
                    kernel_regularizer=regularizers.l2(opts.l2reg)))
    model.add(Dropout(opts.dropout))
    # next hidden layers
    for i in range(1, nhl):
        factor_units = 2 ** (i + 1)
        factor_dropout = 2 * i
        model.add(Dense(units=int(input_size / factor_units),
                        activation=opts.activationFunction,
                        kernel_regularizer=regularizers.l2(opts.l2reg)))
        model.add(Dropout(opts.dropout / factor_dropout))
    # multi-class output layer
    # use sigmoid to get independent probabilities for each output node
    # we use sigmoid since one chemical can be active for multiple targets
    # (need not add up to one, as they would using softmax)
    # https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
    model.add(Dense(units=output_size,
                    activation='sigmoid'))

    model.summary(print_fn=logging.info)

    # compile model
    model.compile(loss="binary_crossentropy",
                  optimizer=my_optimizer,
                  metrics=['accuracy', metrics.Recall(), metrics.Precision()])

    return model


def define_nn_model_multi(input_size: int = 2048,
                          output_size: int = None,
                          l2reg: float = 0.001,
                          dropout: float = 0.2,
                          activation: str = 'relu',
                          optimizer: str = 'Adam',
                          lr: float = 0.001,
                          decay: float = 0.01) -> Model:
    if optimizer == 'Adam':
        my_optimizer = optimizers.Adam(learning_rate=lr, decay=decay)
    elif optimizer == 'SGD':
        my_optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=decay)
    else:
        my_optimizer = optimizer

    nhl = int(math.log2(input_size) / 2 - 1)

    model = Sequential()
    # From input to 1st hidden layer
    model.add(Dense(units=int(input_size / 2),
                    input_dim=input_size,
                    activation=activation,
                    kernel_regularizer=regularizers.l2(l2reg)))
    model.add(Dropout(dropout))
    # next hidden layers
    for i in range(1, nhl):
        factor_units = 2 ** (i + 1)
        factor_dropout = 2 * i
        model.add(Dense(units=int(input_size / factor_units),
                        activation=activation,
                        kernel_regularizer=regularizers.l2(l2reg)))
        model.add(Dropout(dropout / factor_dropout))
    # multi-class output layer
    # use sigmoid to get independent probabilities for each output node
    # (need not add up to one, as they would using softmax)
    # https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
    model.add(Dense(units=output_size,
                    activation='sigmoid'))

    model.summary(print_fn=logging.info)

    # compile model
    model.compile(loss="binary_crossentropy",
                  optimizer=my_optimizer,
                  metrics=['accuracy'])

    return model


def validate_multi_model_on_test_data(x_test: array, checkpoint_path: str, y_test: array,
                                      col_names: list, result_file: str) -> list:
    """
    Validate the multi label model on a test data set.

    :param x_test: The feature matrix of the test data set
    :param checkpoint_path: The path containing the weights of the FNN model
    :param y_test: The outcome matrix of the test data set
    :param col_names: The names of the columns that are targets
    :param result_file: The filename of the output file
    :return: A pandas Dataframe containing the percentage of correct predictions for each target
    """
    # load checkpoint model with min(val_loss)
    trained_model = define_nn_model_multi(input_size=x_test.shape[1],
                                          output_size=y_test.shape[1])

    # predict values with random model
    predictions_random = pd.DataFrame(trained_model.predict(x_test),
                                      columns=[n for n in col_names])

    # load weights into random model
    trained_model.load_weights(checkpoint_path)

    # predict with trained model
    predictions = pd.DataFrame(trained_model.predict(x_test),
                               columns=[n for n in col_names])

    y_true = pd.DataFrame(y_test, columns=col_names)

    f1_random = f1_score(y_true=y_true,
                         y_pred=predictions_random.round(),
                         average='weighted')

    f1_trained = f1_score(y_true=y_true,
                          y_pred=predictions.round(),
                          average='weighted')

    predictions_random.columns = [n + '-random' for n in col_names]
    predictions.columns = [n + '-trained' for n in col_names]

    results = pd.concat([predictions_random, predictions, y_true],
                        axis=1)
    results.to_csv(result_file)

    return [f1_random, f1_trained]


def train_nn_models_multi(df: pd.DataFrame, opts: options.Options) -> None:
    # find target columns
    names_y = [c for c in df.columns if c not in ['cid', 'id', 'smiles', 'fp', 'inchi', 'fpcompressed']]
    selector = df[names_y].notna().apply(np.logical_and.reduce, axis=1)

    if opts.compressFeatures:
        # get compressed fingerprints as numpy array
        fpMatrix = np.array(
            df[df['fpcompressed'].notnull() & selector]['fpcompressed'].to_list(),
            dtype=settings.nn_multi_fp_compressed_numpy_type,
            copy=settings.numpy_copy_values)
        y = np.array(
            df[df['fpcompressed'].notnull() & selector][names_y],
            dtype=settings.nn_multi_target_numpy_type,
            copy=settings.numpy_copy_values)
    else:
        # get fingerprints as numpy array
        fpMatrix = np.array(
            df[df['fp'].notnull() & selector]['fp'].to_list(),
            dtype=settings.nn_multi_fp_numpy_type,
            copy=settings.numpy_copy_values)

        y = np.array(
            df[df['fp'].notnull() & selector][names_y],
            dtype=settings.nn_multi_target_numpy_type,
            copy=settings.numpy_copy_values)

    if opts.kFolds > 0:

        # do a kfold cross validation for the autoencoder training
        kfold_c_validator = KFold(n_splits=opts.kFolds,
                                  shuffle=True,
                                  random_state=42)

        # store acc and loss for each fold
        all_scores = pd.DataFrame(columns=["fold_no",  # fold number of k-fold CV
                                           "loss", "val_loss", "acc", "val_acc",  # FNN training
                                           "f1_random", "f1_trained"])  # F1 scores of predictions

        fold_no = 1

        # split the data
        for train, test in kfold_c_validator.split(fpMatrix, y):
            # kf = kfold_c_validator.split(fpMatrix, y)
            # train, test = next(kf)

            (model_file_path_weights, model_file_path_json, model_hist_path,
             model_hist_csv_path, model_predict_valset_csv_path,
             model_validation, model_auc_file,
             model_auc_file_data, out_file_path, checkpoint_path,
             model_heatmap_x, model_heatmap_z) = define_out_file_names(path_prefix=opts.outputDir,
                                                                       target="multi" + "_compressed-" + str(
                                                                           opts.compressFeatures),
                                                                       fold=fold_no)
            # use a dnn for multi-class prediction
            model = define_nn_model_multi(input_size=fpMatrix[train].shape[1],
                                          output_size=y.shape[1])

            callback_list = cb.nn_callback(checkpoint_path=checkpoint_path, opts=opts)
            # measure the training time
            start = time()

            # train and validate
            hist = model.fit(fpMatrix[train], y[train],
                             callbacks=callback_list,
                             epochs=opts.epochs,
                             batch_size=256,
                             verbose=opts.verbose,
                             validation_split=opts.testSize)

            trainTime = str(round((time() - start) / 60, ndigits=2))

            if opts.verbose > 0:
                logging.info("Computation time for training the multi-label FNN: " + trainTime + " min")

            ht.store_and_plot_history(base_file_name=model_hist_path,
                                      hist=hist)
            # pd.DataFrame(hist.history).to_csv(model_hist_csv_path)

            # validate model on test data set (fpMatrix_test, y_test)
            scores = validate_multi_model_on_test_data(x_test=fpMatrix[test],
                                                       checkpoint_path=checkpoint_path,
                                                       y_test=y[test],
                                                       col_names=names_y,
                                                       result_file=out_file_path.replace("trainingResults.txt",
                                                                                         "predictionResults.csv"))

            idx = hist.history['val_loss'].index(min(hist.history['val_loss']))
            row_df = pd.DataFrame([[fold_no,
                                    hist.history['loss'][idx], hist.history['val_loss'][idx],
                                    hist.history['accuracy'][idx], hist.history['val_accuracy'][idx],
                                    scores[0], scores[1]]],
                                  columns=["fold_no",  # fold number of k-fold CV
                                           "loss", "val_loss", "acc", "val_acc", "f1_random", "f1_trained"]
                                  )

            logging.info(row_df)
            all_scores = all_scores.append(row_df, ignore_index=True)

            fold_no += 1
            del model

        logging.info(all_scores)

        # finalize model
        # 1. provide best performing fold variant
        # select best model based on MCC
        idx2 = all_scores[['f1_trained']].idxmax().ravel()[0]
        fold_no = all_scores.iloc[idx2]['fold_no']

        model_name = "multi" + "_compressed-" + str(opts.compressFeatures) + '.Fold-' + str(fold_no)
        checkpoint_path = opts.outputDir + '/' + model_name + '.checkpoint.model.hdf5'
        best_model_file = checkpoint_path.replace("Fold-" + str(fold_no) + ".checkpoint.", "best.FNN-")

        file = re.sub(".hdf5", "scores.csv", re.sub("Fold-..checkpoint", "Fold-All", checkpoint_path))
        all_scores.to_csv(file)

        # copy best DNN model
        shutil.copyfile(checkpoint_path, best_model_file)
        logging.info("Best models for FNN is saved:\n" + best_model_file)

    # AND retrain with full data set
    # full_model_file = f"{opts.outputDir}/multi_full.FNN.checkpoint.model.hdf5"

    (model_file_path_weights, model_file_path_json, model_hist_path, model_hist_csv_path,
     model_predict_valset_csv_path,
     model_validation, model_auc_file,
     model_auc_file_data, out_file_path, checkpoint_path,
     model_heatmap_x, model_heatmap_z) = define_out_file_names(path_prefix=opts.outputDir,
                                                               target="multi" + "_compressed-" + str(
                                                                   opts.compressFeatures))

    # test training data split
    X_train, X_test, y_train, y_test = train_test_split(fpMatrix, y, test_size=opts.testSize)

    # measure the training time
    model = define_nn_multi_label_model(input_size=fpMatrix.shape[1],
                                        output_size=y.shape[1],
                                        opts=opts)
    model.evaluate(X_test, y_test)

    # model = define_nn_model_multi(input_size=fpMatrix.shape[1],
    #                               output_size=y.shape[1])
    # callback_list = nn_callback(checkpoint_path=full_model_file, opts=opts)
    # train and validate
    start = time()
    hist = model.fit(X_train, y_train,
                     # callbacks=callback_list,
                     epochs=opts.epochs,
                     batch_size=opts.batchSize,
                     verbose=opts.verbose,
                     validation_data=(X_test, y_test)
                     )
    trainTime = str(round((time() - start) / 60, ndigits=2))

    # yhat = model.predict(X_test)
    model.evaluate(X_test, y_test)

    logging.info("Computation time for training the full multi-label FNN: " + trainTime + " min")
    ht.store_and_plot_history(base_file_name=model_hist_path, hist=hist)
