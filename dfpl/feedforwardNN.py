import array
import re
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import logging
import shutil

# for NN model functions
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras import optimizers
from keras import metrics
from keras.callbacks import History, ModelCheckpoint, EarlyStopping

from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

from dfpl import options
from dfpl import autoencoder as ac
from dfpl import history as ht
from dfpl import settings

from time import time

# for testing in Weights & Biases
import wandb
from wandb.keras import WandbCallback

def define_nn_single_label_model(input_size: int,
                                 opts: options.TrainOptions) -> Model:
    lf = dict({"mse": "mean_squared_error",
               "bce": "binary_crossentropy"})

    if opts.optimizer == 'Adam':
        my_optimizer = optimizers.Adam(learning_rate=opts.learningRate)
    elif opts.optimizer == 'SGD':
        my_optimizer = optimizers.SGD(lr=opts.learningRate, momentum=0.9)
    else:
        logging.error(f"Your selected optimizer is not supported:{opts.optimizer}.")
        sys.exit("Unsupported optimizer.")

    my_hidden_layers = {"2048": 6, "1024": 5, "999": 5, "512": 4, "256": 3}

    if not str(input_size) in my_hidden_layers.keys():
        raise ValueError("Wrong input-size. Must be in {2048, 1024, 999, 512, 256}.")

    nhl = int(math.log2(input_size) / 2 - 1)

    model = Sequential()
    # From input to 1st hidden layer
    model.add(Dense(units=int(input_size / 2),
                    input_dim=input_size,
                    activation=opts.activationFunction,
                    kernel_regularizer=regularizers.l2(opts.l2reg),
                    kernel_initializer="he_uniform"))
    model.add(Dropout(opts.dropout))
    # next hidden layers
    for i in range(1, nhl):
        factor_units = 2 ** (i + 1)
        factor_dropout = 2 * i
        model.add(Dense(units=int(input_size / factor_units),
                        activation=opts.activationFunction,
                        kernel_regularizer=regularizers.l2(opts.l2reg),
                        kernel_initializer="he_uniform"))
        model.add(Dropout(opts.dropout / factor_dropout))
    # output layer
    model.add(Dense(units=1,
                    activation='sigmoid'))

    model.summary(print_fn=logging.info)

    # compile model
    model.compile(loss=lf[opts.lossFunction],
                  optimizer=my_optimizer,
                  metrics=['accuracy',
                           metrics.Precision(),
                           metrics.Recall()]
                  )
    return model


def define_nn_model(
        input_size: int = 2048,
        l2reg: float = 0.001,
        dropout: float = 0.2,
        activation: str = 'relu',
        optimizer: str = 'Adam',
        lr: float = 0.001,
        decay: float = 0.01,
        loss_function: str = "mse") -> Model:
    """
    Sets up the structure of the feed forward neural network. The number and size of the hidden layers are based on
    the dimensions of the input vector.

    :param loss_function: The loss function used during training. Values: mse or bco. Default 'mse'.
    :param input_size: Length of the input vector. Default: 2048
    :param l2reg: Log2 regularization value. Default: 0.001
    :param dropout: Value of dropout for hidden layers. Default: 0.2
    :param activation: Activation function for inner layers. Default: 'relu'
    :param optimizer: Optimizer for loss function. Default: 'Adam'
    :param lr: Learning rate. Default: 0.001
    :param decay: Decay of the optimizer. Default: 0.01
    :return: A keras model.
    """

    lf = dict({"mse": "mean_squared_error",
               "bce": "binary_crossentropy"})

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
                    kernel_regularizer=regularizers.l2(l2reg),
                    kernel_initializer="he_uniform"))
    model.add(Dropout(dropout))
    # next hidden layers
    for i in range(1, nhl):
        factor_units = 2 ** (i + 1)
        factor_dropout = 2 * i
        model.add(Dense(units=int(input_size / factor_units),
                        activation=activation,
                        kernel_regularizer=regularizers.l2(l2reg),
                        kernel_initializer="he_uniform"))
        model.add(Dropout(dropout / factor_dropout))
    # output layer
    model.add(Dense(units=1,
                    activation='sigmoid'))

    model.summary(print_fn=logging.info)

    # compile model
    model.compile(loss=lf[loss_function],
                  optimizer=my_optimizer,
                  metrics=['accuracy',
                           metrics.Precision(),
                           metrics.Recall()]
                  )  # ,
    # metrics.Accuracy(),
    # metrics.Precision(),
    # metrics.Recall(),
    # metrics.SpecificityAtSensitivity(),
    # metrics.SensitivityAtSpecificity()])

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


def prepare_nn_training_data(df: pd.DataFrame, target: str, opts: options.TrainOptions) -> (np.ndarray,
                                                                                            np.ndarray,
                                                                                            options.TrainOptions):
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
        if opts.sampleFractionOnes:
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


def nn_callback(checkpoint_path: str) -> list:
    """
    Callbacks for fitting the autoencoder

    :param checkpoint_path: The output directory to store the checkpoint weight files
    :return: List of ModelCheckpoint and EarlyStopping class.
    """

    # enable this checkpoint to restore the weights of the best performing model
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 verbose=1,
                                 period=settings.nn_train_check_period,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)

    # enable early stopping if val_loss is not improving anymore
    early_stop = EarlyStopping(patience=settings.nn_train_patience,
                               min_delta=settings.nn_train_min_delta,
                               verbose=1,
                               restore_best_weights=True)

    trackWandB_callback = WandbCallback()
    # trackWandB_callback = WandbCallback(monitor={'Train loss': 'loss', 'Val loss': 'val_loss',
    #                                              'Train accuracy': 'accuracy', 'Val accuracy': 'val_accuracy'})

    return [checkpoint, early_stop, trackWandB_callback]


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
    names_y = [c for c in df.columns if c not in ['cid', 'id', 'mol_id', 'smiles', 'fp', 'inchi', 'fpcompressed']]

    # For each individual target train a model
    for target in names_y:  # [:1]:
        # target=names_y[1] # --> only for testing the code
        if opts.trackWandB:
            wandb.init(project=f"dfpl-training-{target}", config=vars(opts))
            opts = wandb.config

        x, y, opts = prepare_nn_training_data(df, target, opts)
        if x is None:
            continue

        logging.info(f"X training matrix of shape {x.shape} and type {x.dtype}")
        logging.info(f"Y training matrix of shape {y.shape} and type {y.dtype}")

        if opts.kFolds > 0:
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

                logging.info("Training of fold number:" + str(fold_no))

                logging.info(f"The distribution of 0 and 1 values is:")
                logging.info(f"\ttrain data:\t{pd.DataFrame(y[train])[0].value_counts().to_list()}")
                logging.info(f"\ttest  data:\t{pd.DataFrame(y[test])[0].value_counts().to_list()}")

                model_name = target + "_compressed-" + str(opts.compressFeatures) + "_sampled-" + \
                             str(opts.sampleFractionOnes)

                # define all the output file/path names
                (model_file_path_weights, model_file_path_json, model_hist_path, model_hist_csv_path,
                 model_predict_valset_csv_path,
                 model_validation, model_auc_file,
                 model_auc_file_data, outfile_path, checkpoint_path,
                 model_heatmap_x, model_heatmap_z) = define_out_file_names(path_prefix=opts.outputDir,
                                                                           target=model_name,
                                                                           fold=fold_no)

                model = define_nn_model(input_size=x[train].shape[1],
                                        loss_function=opts.lossFunction,
                                        optimizer=opts.optimizer)

                callback_list = nn_callback(checkpoint_path=checkpoint_path)

                # measure the training time
                start = time()
                # train and validate
                # y_int_train = np.array([int(i) for i in y[train]])
                # y_int_test = np.array([int(i) for i in y[test]])
                hist = model.fit(x[train], y[train],  # y_int_train,
                                 callbacks=callback_list,
                                 epochs=opts.epochs,
                                 batch_size=opts.batchSize,
                                 verbose=opts.verbose,
                                 # validation_split=opts.testSize
                                 validation_data=(x[test], y[test])  # y_int_test)  # this overwrites val_split!
                                 )
                trainTime = str(round((time() - start) / 60, ndigits=2))

                logging.info("Computation time for training the single-label FNN:" + trainTime + "min")

                # ht.store_and_plot_history(base_file_name=model_hist_path,
                #                           hist=hist)

                pd.DataFrame(hist.history).to_csv(model_hist_csv_path)

                # predict test data set (x_test, y_test) and store it to generate precision recall AUC curves
                y_predict = model.predict(x[test]).flatten()
                pd.DataFrame({"y_true": y[test], "y_predicted": y_predict}).to_csv(model_predict_valset_csv_path)

                # scores = validate_model_on_test_data(x[test], checkpoint_path, y[test],
                #                                      "FNN", model_validation, target,
                #                                      model_auc_file_data, model_auc_file)
                #
                # idx = hist.history['val_loss'].index(min(hist.history['val_loss']))
                #
                # row_df = pd.DataFrame([[fold_no,
                #                         hist.history['loss'][idx], hist.history['val_loss'][idx],
                #                         hist.history['accuracy'][idx], hist.history['val_accuracy'][idx],
                #                         scores[0], scores[1], scores[2]]],
                #                       columns=["fold_no",  # fold number of k-fold CV
                #                                "loss", "val_loss", "acc", "val_acc",  # FNN training
                #                                "loss_test", "acc_test", "mcc_test"]
                #                       )
                # logging.info(row_df)
                # all_scores = all_scores.append(row_df, ignore_index=True)
                fold_no += 1
                del model
                # now next fold

            logging.info(all_scores)

            # finalize model
            # 1. provide best performing fold variant
            # select best model based on MCC
            # idx2 = all_scores[['mcc_test']].idxmax().ravel()[0]
            # fold_no = all_scores.iloc[idx2]['fold_no']
            #
            # model_name = target + "_compressed-" + str(opts.compressFeatures) + "_sampled-" + str(
            #     opts.sampleFractionOnes) + '.Fold-' + str(fold_no)
            # checkpoint_path = str(opts.outputDir) + "/" + model_name + '.checkpoint.model.hdf5'
            #
            # best_model_file = checkpoint_path.replace("Fold-" + str(fold_no) + ".checkpoint", "best.FNN")
            #
            # # store all scores
            # file = re.sub(".hdf5", "scores.csv", re.sub("Fold-..checkpoint", "Fold-All", checkpoint_path))
            # all_scores.to_csv(file)
            #
            # # copy best DNN model
            # shutil.copyfile(checkpoint_path, best_model_file)
            # logging.info("Best model for FNN is saved: " + best_model_file)

        # train with full data set
        # test training data split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=opts.testSize)

        full_model_file = f"{opts.outputDir}/{target}_full.FNN.model.hdf5"
        (model_file_path_weights, model_file_path_json, model_hist_path,
         model_hist_csv_path, model_predict_valset_csv_path,
         model_validation, model_auc_file,
         model_auc_file_data, out_file_path, checkpoint_path,
         model_heatmap_x, model_heatmap_z) = define_out_file_names(path_prefix=opts.outputDir,
                                                                   target=f"{target}"
                                                                          f"_compressed-{str(opts.compressFeatures)}"
                                                                          f"_sampled-{str(opts.sampleFractionOnes)}")

        # model = define_nn_model(input_size=x.shape[1])
        model = define_nn_single_label_model(input_size=X_train.shape[1], opts=opts)
        # evaluate randomly initiated model
        performance = model.evaluate(X_test, y_test)
        logging.info(f"{target}-model (random) evaluation [loss, accuracy, precision, recall]:\n"
                     f"{[round(i, ndigits=4) for i in performance]}")

        callback_list = nn_callback(checkpoint_path=full_model_file)

        # measure the training time
        start = time()
        # train and validate
        hist = model.fit(X_train, y_train,
                         callbacks=callback_list,
                         epochs=opts.epochs,
                         batch_size=opts.batchSize,
                         verbose=opts.verbose,
                         validation_data=(X_test, y_test)
                         )

        trainTime = str(round((time() - start) / 60, ndigits=2))

        y_predict = model.predict(X_test).flatten()
        pd.DataFrame({"y_true": y_test, "y_predicted": y_predict}).to_csv(model_predict_valset_csv_path)
        performance = model.evaluate(X_test, y_test)

        logging.info(f"Computation time for training the full classification FNN: {trainTime}min")
        logging.info(f"{target}-model (trained) evaluation [loss, accuracy, precision, recall]:\n"
                     f"{[round(i, ndigits=4) for i in performance]}")

        model_hist_path = full_model_file.replace(".hdf5", "")
        # ht.store_and_plot_history(base_file_name=model_hist_path,
        #                           hist=hist)

        # pd.DataFrame(hist.history).to_csv(full_model_file.replace(".hdf5", ".history.csv"))
        pd.DataFrame(hist.history).to_csv(model_hist_csv_path)

        del model
        # now next target


def define_nn_multi_label_model(input_size: int,
                                output_size: int,
                                opts: options.TrainOptions) -> Model:
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
             model_validation, model_auc_file,
             model_auc_file_data, out_file_path, checkpoint_path,
             model_heatmap_x, model_heatmap_z) = define_out_file_names(path_prefix=opts.outputDir,
                                                                       target="multi" + "_compressed-" + str(
                                                                           opts.compressFeatures),
                                                                       fold=fold_no)

            # use a dnn for multi-class prediction
            model = define_nn_model_multi(input_size=fpMatrix[train].shape[1],
                                          output_size=y.shape[1])

            callback_list = nn_callback(checkpoint_path=checkpoint_path)
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
    full_model_file = f"{opts.outputDir}/multi_full.FNN.checkpoint.model.hdf5"

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
    callback_list = nn_callback(checkpoint_path=full_model_file)
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
