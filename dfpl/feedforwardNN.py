import array
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import logging
import shutil
from os import path

# for NN model functions
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras import optimizers
from keras.optimizers import SGD
from keras.callbacks import History

from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import f1_score

import options
import autoencoder as ac
import history as ht

from time import time


def define_nn_model(
        input_size: int = 2048,
        l2reg: float = 0.001,
        dropout: float = 0.2,
        activation: str = 'relu',
        optimizer: str = 'Adam',
        lr: float = 0.001,
        decay: float = 0.01) -> Model:
    """
    Sets up the structure of the feed forward neural network. The number and size of the hidden layers are based on
    the dimensions of the input vector.

    :param input_size: Length of the input vector. Default: 2048
    :param l2reg: Log2 regularization value. Default: 0.001
    :param dropout: Value of dropout for hidden layers. Default: 0.2
    :param activation: Activation function for inner layers. Default: 'relu'
    :param optimizer: Optimizer for loss function. Default: 'Adam'
    :param lr: Learning rate. Default: 0.001
    :param decay: Decay of the optimizer. Default: 0.01
    :return: A keras model.
    """

    if optimizer == 'Adam':
        my_optimizer = optimizers.Adam(learning_rate=lr,
                                       decay=decay)
    elif optimizer == 'SGD':
        my_optimizer = SGD(lr=lr,
                           momentum=0.9,
                           decay=decay)
    else:
        my_optimizer = optimizer

    my_hidden_layers = {"2048": 6, "1024": 5, "999": 5, "512": 4, "256": 3}

    if not str(input_size) in my_hidden_layers.keys():
        raise ValueError("Wrong input-size. Must be in {2048, 1024, 999, 512, 256}.")

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
    # output layer
    model.add(Dense(units=1,
                    activation='sigmoid'))

    model.summary(print_fn=logging.info)

    # compile model
    model.compile(loss="mse",
                  optimizer=my_optimizer,
                  metrics=['accuracy'])

    return model


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
    model_validation = str(path_prefix) + model_name + '.validation.csv'
    model_auc_file = str(path_prefix) + model_name + '.auc_value.svg'
    model_auc_file_data = str(path_prefix) + model_name + '.auc_value.data.csv'
    out_file_path = str(path_prefix) + model_name + '.trainingResults.txt'
    checkpoint_path = str(path_prefix) + model_name + '.checkpoint.model.hdf5'
    model_heatmap_x = str(path_prefix) + model_name + '.heatmap.X.svg'
    model_heatmap_z = str(path_prefix) + model_name + '.AC.heatmap.Z.svg'

    return (model_file_path_weights, model_file_path_json, model_hist_path,
            model_validation, model_auc_file,
            model_auc_file_data, out_file_path, checkpoint_path,
            model_heatmap_x, model_heatmap_z)


def get_max_validation_accuracy(history: History) -> str:
    validation = smooth_curve(history.history['val_accuracy'])
    y_max = max(validation)
    return 'Max validation accuracy ≈ ' + str(round(y_max, 3) * 100) + '%'


def get_max_training_accuracy(history: History) -> str:
    training = smooth_curve(history.history['accuracy'])
    y_max = max(training)
    return 'Max training accuracy ≈ ' + str(round(y_max, 3) * 100) + '%'


def smooth_curve(points: array, factor: float = 0.75) -> array:
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def set_plot_history_data(ax: Axes, history: History, which_graph: str) -> None:
    (train, valid) = (None, None)

    if which_graph == 'acc':
        train = smooth_curve(history.history['accuracy'])
        valid = smooth_curve(history.history['val_accuracy'])

    if which_graph == 'loss':
        train = smooth_curve(history.history['loss'])
        valid = smooth_curve(history.history['val_loss'])

    # plt.xkcd() # make plots look like xkcd

    epochs = range(1, len(train) + 1)

    trim = 0  # remove first 5 epochs
    # when graphing loss the first few epochs may skew the (loss) graph

    ax.plot(epochs[trim:], train[trim:], 'dodgerblue', linewidth=15, alpha=0.1)
    ax.plot(epochs[trim:], train[trim:], 'dodgerblue', label='Training')

    ax.plot(epochs[trim:], valid[trim:], 'g', linewidth=15, alpha=0.1)
    ax.plot(epochs[trim:], valid[trim:], 'g', label='Validation')


def plot_history(history: History, file: str) -> None:
    fig, (ax1, ax2) = plt.subplots(nrows=2,
                                   ncols=1,
                                   figsize=(10, 6),
                                   sharex='all',
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


def plotTrainHistory(hist, target, file_accuracy, file_loss):
    """
    Plot the training performance in terms of accuracy and loss values for each epoch.
    :param hist: The history returned by model.fit function
    :param target: The name of the target of the model
    :param file_accuracy: The filename for plotting accuracy values
    :param file_loss: The filename for plotting loss values
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
    plt.savefig(fname=file_accuracy, format='svg')

    # Plot training & validation loss values
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss - ' + target)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    #        plt.show()
    plt.savefig(fname=file_loss, format='svg')
    plt.close()


def plot_history_vis(hist: History, model_hist_plot_path: str, model_hist_csv_path: str,
                     model_hist_plot_path_a: str, model_hist_plot_path_l: str, target: str) -> None:
    plot_history(history=hist, file=model_hist_plot_path)
    histDF = pd.DataFrame(hist.history)
    histDF.to_csv(model_hist_csv_path)

    # plot accuracy and loss for the training and validation during training
    plotTrainHistory(hist=hist, target=target,
                     file_accuracy=model_hist_plot_path_a,
                     file_loss=model_hist_plot_path_l)


def plot_auc(fpr: array, tpr: array, auc_value: float, target: str, filename: str) -> None:
    """
    Plot the area under the curve to the provided file

    :param fpr: An array containing the false positives
    :param tpr: An array containing the true positives
    :param auc_value: The value of the area under the curve
    :param target: The name of the training target
    :param filename: The filename to which the plot should be stored
    :rtype: None
    """
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_value))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve ' + target)
    plt.legend(loc='best')
    plt.savefig(fname=filename, format='svg')
    plt.close()


def validate_model_on_test_data(x_test: array, checkpoint_path: str, y_test: array, model_type: str,
                                model_validation: str, target: str, model_auc_file_data: str,
                                model_auc_file: str) -> tuple:
    """
    Function that validates trained model with test data set. History and AUC plots are generated.
    Accuracy and Loss of model on test data, as well as MCC and confusion matrix is calculated and returned.

    :param x_test:
    :param checkpoint_path:
    :param y_test:
    :param model_type:
    :param model_validation:
    :param target:
    :param model_auc_file_data:
    :param model_auc_file:
    :return: Tuple containing Loss, Accuracy, MCC, tn, fp, fn, tp of trained model on test data.
    """

    # load checkpoint model with min(val_loss)
    trained_model = define_nn_model(input_size=x_test.shape[1])

    # predict values with random model
    predictions_random = pd.DataFrame(trained_model.predict(x_test))

    # load weights into random model
    trained_model.load_weights(checkpoint_path)
    # predict with trained model
    predictions = pd.DataFrame(trained_model.predict(x_test))

    # save validation data to .csv file
    validation = pd.DataFrame({'predicted': predictions[0].ravel(),
                               'true': list(y_test),
                               'predicted_random': predictions_random[0].ravel(),
                               'model_type': model_type})
    validation.to_csv(model_validation)

    # compute MCC
    predictions_int = [int(round(x)) for x in predictions[0].ravel()]
    y_true_int = [int(y) for y in y_test]
    MCC = matthews_corrcoef(y_true_int, predictions_int)

    # generate the AUC-ROC curve data from the validation data
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true=y_test,
                                                       y_score=predictions,
                                                       drop_intermediate=False)

    auc_keras = auc(fpr_keras, tpr_keras)

    auc_data = pd.DataFrame(list(zip(fpr_keras,
                                     tpr_keras,
                                     [auc_keras] * len(fpr_keras),
                                     [target] * len(fpr_keras),
                                     )),
                            columns=['fpr', 'tpr', 'auc_value', 'target'])
    auc_data.to_csv(model_auc_file_data)

    plot_auc(fpr=fpr_keras,
             tpr=tpr_keras,
             target=target,
             auc_value=auc_keras,
             filename=model_auc_file)

    # [[tn, fp]
    #  [fn, tp]]
    cfm = confusion_matrix(y_true=y_true_int,
                           y_pred=predictions_int)

    scores = trained_model.evaluate(x=x_test,
                                    y=y_test,
                                    verbose=0)

    logging.info('TARGET: ' + target + 'Loss: ' + str(scores[0].__round__(2)) + 'Acc: ' + str(scores[1].__round__(2)))
    logging.info('MCC: ' + str(MCC.__round__(2)))
    logging.info('CFM: \n\tTN=' + str(cfm[0][0]) + '\tFP=' + str(cfm[0][1]) + '\n\tFN=' + str(cfm[1][0]) +
                 '\tTP=' + str(cfm[1][1]))

    return scores[0], scores[1], MCC, cfm[0][0], cfm[0][1], cfm[1][0], cfm[1][1]


def prepare_nn_training_data(df: pd.DataFrame, target: str, opts: options.TrainOptions) -> (np.ndarray, np.ndarray):
    if opts.compressFeatures:
        df_fpc = df[df[target].notna() & df["fpcompressed"].notnull()]
        if opts.sampleFractionOnes:
            # how many ones
            counts = df_fpc[target].value_counts()

            # add sample of 0s to df of 1s
            dfX = df_fpc[df_fpc[target] == 1].append(
                df_fpc[df_fpc[target] == 0].sample(
                    # round(min(counts[0], counts[1] / opts.sampleFractionOnes))
                    int(min(counts[0], counts[1] / opts.sampleFractionOnes))
                )
            )
            x = np.array(dfX["fpcompressed"].to_list(), dtype=bool, copy=False)
            y = np.array(dfX[target].to_list(), copy=False)
        else:
            x = np.array(df_fpc["fpcompressed"].to_list(), dtype=bool, copy=False)
            y = np.array(df_fpc[target].to_list(), copy=False)
    else:
        df_fp = df[df[target].notna() & df["fp"].notnull()]
        if opts.sampleFractionOnes:
            counts = df_fp[target].value_counts()
            dfX = df_fp[df_fp[target] == 1.0].append(
                df_fp[df_fp[target] == 0.0].sample(
                    round(min(counts[0], counts[1] / opts.sampleFractionOnes)))
            )
            x = np.array(dfX["fp"].to_list(), dtype=bool, copy=False)
            y = np.array(dfX[target].to_list(), copy=False)
        else:
            x = np.array(df_fp["fp"].to_list())
            y = np.array(df_fp[target].to_list())
    return x, y


def train_nn_models(df: pd.DataFrame, opts: options.TrainOptions) -> None:
    """
    Train individual models for all targets (columns) present in the provided target data (y) and a multi-label
    model that classifies all targets at once. For each individual target the data is first subset to exclude NA
    values (for target associations). A random sample of the remaining data (size is the split fraction) is used for
    training and the remaining data for validation.

    :param opts: The command line arguments in the options class
    :param df: The dataframe containing x matrix and at least one column for a y target.
    """

    # find target columns
    names_y = [c for c in df.columns if c not in ['id', 'smiles', 'fp', 'inchi', 'fpcompressed']]

    # For each individual target train a model
    for target in names_y:  # [:2]:
        # target=names_y[0] # --> only for testing the code

        x, y = prepare_nn_training_data(df, target, opts)

        # do a kfold cross validation for the FNN training
        kfold_c_validator = KFold(n_splits=opts.kFolds,
                                  shuffle=True,
                                  random_state=42)

        # store acc and loss for each fold
        all_scores = pd.DataFrame(columns=["fold_no",  # fold number of k-fold CV
                                           "loss", "val_loss", "acc", "val_acc",  # FNN training
                                           "loss_test", "acc_test"])  # FNN test data

        fold_no = 1

        # split the data
        for train, test in kfold_c_validator.split(x, y):  # kfold_c_validator.split(Xt, Yt):
            # for testing use one of the splits:
            # kf = kfold_c_validator.split(x, y)
            # train, test = next(kf)

            if opts.verbose > 0:
                logging.info("Training of fold number:" + str(fold_no))

            model_name = target + "_compressed-" + str(opts.compressFeatures) + "_sampled-" + str(opts.sampleFractionOnes)

            # define all the output file/path names
            (model_file_path_weights, model_file_path_json, model_hist_path,
             model_validation, model_auc_file,
             model_auc_file_data, outfile_path, checkpoint_path,
             model_heatmap_x, model_heatmap_z) = define_out_file_names(path_prefix=opts.outputDir,
                                                                       target=model_name,
                                                                       fold=fold_no)

            model = define_nn_model(input_size=x[train].shape[1])

            callback_list = ac.autoencoder_callback(checkpoint_path=checkpoint_path,
                                                    patience=20)

            # measure the training time
            start = time()
            # train and validate
            hist = model.fit(x[train], y[train],
                             callbacks=callback_list,
                             epochs=opts.epochs,
                             batch_size=256,
                             verbose=opts.verbose,
                             validation_split=opts.testingFraction)
            #                             validation_data=(x_test, y_test))  # this overwrites val_split!
            trainTime = str(round((time() - start) / 60,
                                  ndigits=2))

            if opts.verbose > 0:
                logging.info("Computation time for training the single-label FNN:" + trainTime + "min")

            ht.store_and_plot_history(base_file_name=model_hist_path,
                                      hist=hist)

            # pd.DataFrame(hist.history).to_csv(model_hist_csv_path)

            # validate model on test data set (x_test, y_test)
            scores = validate_model_on_test_data(x[test], checkpoint_path, y[test],
                                                 "FNN", model_validation, target,
                                                 model_auc_file_data, model_auc_file)

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
            all_scores = all_scores.append(row_df, ignore_index=True)
            fold_no += 1
            del model
            # now next fold

        print(all_scores)

        # finalize model
        # 1. provide best performing fold variant
        # select best model based on MCC
        idx2 = all_scores[['mcc_test']].idxmax().ravel()[0]
        fold_no = all_scores.iloc[idx2]['fold_no']

        model_name = target + "_compressed-" + str(opts.compressFeatures) + "_sampled-" + str(
            opts.sampleFractionOnes) + '.Fold-' + str(fold_no)
        checkpoint_path = str(opts.outputDir) + "/" + model_name + '.checkpoint.model.hdf5'

        best_model_file = checkpoint_path.replace("Fold-" + str(fold_no) + ".checkpoint", "best.FNN")

        # store all scores
        file = re.sub(".hdf5", "scores.csv", re.sub("Fold-..checkpoint", "Fold-All", checkpoint_path))
        all_scores.to_csv(file)

        # copy best DNN model
        shutil.copyfile(checkpoint_path, best_model_file)
        logging.info("Best model for FNN is saved: " + best_model_file)

        # AND retrain with full data set
        full_model_file = checkpoint_path.replace("Fold-" + str(fold_no) + ".checkpoint", "full.FNN")
        (model_file_path_weights, model_file_path_json, model_hist_path,
         model_validation, model_auc_file,
         model_auc_file_data, out_file_path, checkpoint_path,
         model_heatmap_x, model_heatmap_z) = define_out_file_names(path_prefix=opts.outputDir,
                                                                   target=target + "_compressed-" + str(
                                                                       opts.compressFeatures) + "_sampled-" + str(
                                                                       opts.sampleFractionOnes))
        # measure the training time
        start = time()

        model = define_nn_model(input_size=x.shape[1])
        callback_list = ac.autoencoder_callback(checkpoint_path=full_model_file,
                                                patience=20)

        # train and validate
        hist = model.fit(x, y,
                         callbacks=callback_list,
                         epochs=opts.epochs,
                         batch_size=256,
                         verbose=opts.verbose,
                         validation_split=opts.testingFraction)

        trainTime = str(round((time() - start) / 60,
                              ndigits=2))

        if opts.verbose > 0:
            logging.info("Computation time for training the full classification FNN: " + trainTime + "min")

        model_hist_path = full_model_file.replace(".hdf5", "")
        ht.store_and_plot_history(base_file_name=model_hist_path,
                                  hist=hist)

        # pd.DataFrame(hist.history).to_csv(full_model_file.replace(".hdf5", ".history.csv"))

        del model
        # now next target


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
        my_optimizer = SGD(lr=lr, momentum=0.9, decay=decay)
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


def train_nn_models_multi(df: pd.DataFrame, opts: options.TrainOptions) -> None:
    # find target columns
    names_y = [c for c in df.columns if c not in ['id', 'smiles', 'fp', 'inchi', 'fpcompressed']]
    selector = df[names_y].notna().apply(np.logical_and.reduce, axis=1)

    if opts.compressFeatures:
        # get compressed fingerprints as numpy array
        fpMatrix = np.array(
            df[df['fpcompressed'].notnull() & selector]['fpcompressed'].to_list(),
            dtype=np.bool,
            copy=False)
        y = np.array(
            df[df['fpcompressed'].notnull() & selector][names_y],
            copy=False)
    else:
        # get fingerprints as numpy array
        fpMatrix = np.array(
            df[df['fp'].notnull() & selector]['fp'].to_list(),
            dtype=np.bool,
            copy=False)

        y = np.array(
            df[df['fp'].notnull() & selector][names_y],
            copy=False)

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
         model_validation, model_auc_file,
         model_auc_file_data, out_file_path, checkpoint_path,
         model_heatmap_x, model_heatmap_z) = define_out_file_names(path_prefix=opts.outputDir,
                                                                   target="multi" + "_compressed-" + str(
                                                                       opts.compressFeatures),
                                                                   fold=fold_no)

        # use a dnn for multi-class prediction
        model = define_nn_model_multi(input_size=fpMatrix[train].shape[1],
                                      output_size=y.shape[1])

        callback_list = ac.autoencoder_callback(checkpoint_path=checkpoint_path,
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

        print(row_df)
        all_scores = all_scores.append(row_df, ignore_index=True)

        fold_no += 1
        del model

    print(all_scores)

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
    full_model_file = checkpoint_path.replace("Fold-" + str(fold_no) + ".checkpoint", "full.FNN-")

    (model_file_path_weights, model_file_path_json, model_hist_path,
     model_validation, model_auc_file,
     model_auc_file_data, out_file_path, checkpoint_path,
     model_heatmap_x, model_heatmap_z) = define_out_file_names(path_prefix=opts.outputDir,
                                                               target="multi" + "_compressed-" + str(
                                                                   opts.compressFeatures))

    # measure the training time
    start = time()

    model = define_nn_model_multi(input_size=fpMatrix.shape[1],
                                  output_size=y.shape[1])
    callback_list = ac.autoencoder_callback(checkpoint_path=full_model_file,
                                            patience=20)
    # train and validate
    hist = model.fit(fpMatrix, y,
                     callbacks=callback_list,
                     epochs=opts.epochs,
                     batch_size=256,
                     verbose=opts.verbose,
                     validation_split=opts.testingFraction)

    trainTime = str(round((time() - start) / 60,
                          ndigits=2))

    if opts.verbose > 0:
        logging.info("Computation time for training the full multi-label FNN: " + trainTime + " min")

    ht.store_and_plot_history(base_file_name=model_hist_path,
                              hist=hist)

    # pd.DataFrame(hist.history).to_csv(model_hist_csv_path)

    # model_name = "multi" + "_compressed-" + str(use_compressed) + '.Full'
    #
    # plot_history_vis(hist,
    #                  model_hist_plot_path.replace("Fold-" + str(fold_no), "full.DNN-model"),
    #                  model_hist_csv_path.replace("Fold-" + str(fold_no), "full.DNN-model"),
    #                  model_hist_plot_path_acc.replace("Fold-" + str(fold_no), "full.DNN-model"),
    #                  model_hist_plot_path_loss.replace("Fold-" + str(fold_no), "full.DNN-model"),
    #                  target=model_name)
    # logging.info("Full models for DNN is saved:\n" + full_model_file)

    # pd.DataFrame(hist.history).to_csv(full_model_file.replace(".hdf5", ".history.csv"))
