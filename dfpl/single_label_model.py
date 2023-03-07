import logging
import math
import sys
from os import path
from time import time
import numpy as np
import pandas as pd
import shutil
from typing import Sequence, Tuple, Set
from sklearn.metrics import auc, classification_report, confusion_matrix, matthews_corrcoef, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf
from tensorflow.keras import metrics, optimizers, regularizers
from tensorflow.keras.layers import Dense, Dropout, AlphaDropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import tensorflow.keras.backend as K
from dfpl import callbacks as cb
from dfpl import options
from dfpl import plot as pl
from dfpl import settings
from dfpl.utils import scaffold_split
import wandb


def prepare_nn_training_data(df: pd.DataFrame, target: str, opts: options.Options) -> (np.ndarray, np.ndarray):
    # Check the value counts and abort if too imbalanced
    allowed_imbalance = 0.1

    # If feature compression is enabled, use compressed fingerprints
    if opts.compressFeatures:
        vc = df[df[target].notna() & df["fpcompressed"].notnull()][target].value_counts()
    # Otherwise, use uncompressed fingerprints
    else:
        vc = df[df[target].notna() & df["fp"].notnull()][target].value_counts()

    # If the dataset is extremely unbalanced, adjust sampling options
    if min(vc) < max(vc) * allowed_imbalance:
        logging.info(f" Your training data is extremely unbalanced ({target}): 0 - {vc[0]}, and 1 - {vc[1]} values.")
        if opts.sampleDown:
            logging.info(f" I will down-sample your data")
            opts.sampleFractionOnes = allowed_imbalance
        else:
            logging.info(f" I will not down-sample your data automatically.")
            logging.info(f" Consider to enable down sampling of the 0 values with --sampleDown option.")

    logging.info("Preparing training data matrices")

    # If feature compression is enabled, use compressed fingerprints
    if opts.compressFeatures:
        logging.info("Using compressed fingerprints")
        df_fpc = df[df[target].notna() & df["fpcompressed"].notnull()]
        logging.info(f"DataSet has {df_fpc.shape[0]} valid entries in fpcompressed and {target}")

        # If sampling down is enabled, adjust sampling options and downsample the data
        if opts.sampleDown:
            assert 0.0 < opts.sampleFractionOnes < 1.0
            logging.info(f"Using fractional sampling {opts.sampleFractionOnes}")
            # How many ones and zeros are in the dataset
            counts = df_fpc[target].value_counts()
            logging.info(f"Number of sampling values: {counts.to_dict()}")

            # Add a sample of 0s to the df of 1s
            dfX = df_fpc[df_fpc[target] == 1].append(
                df_fpc[df_fpc[target] == 0].sample(
                    int(min(counts[0], counts[1] / opts.sampleFractionOnes))
                )
            )
            # Convert fingerprints and targets to numpy arrays
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
            # How many ones and zeros are in the dataset
            counts = df_fpc[target].value_counts()
            logging.info(f"Number of sampling values: {counts.to_dict()}")

            # Convert fingerprints and targets to numpy arrays
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
    # If feature compression is not enabled, use uncompressed fingerprints
    else:
        logging.info("Using uncompressed fingerprints")
        # select rows where the target is not null and the fingerprint is not null
        df_fp = df[df[target].notna() & df["fp"].notnull()]
        logging.info(f"DataSet has {df_fp.shape[0]} valid entries in fp and {target}")

        if opts.sampleDown:
            logging.info(f"Using fractional sampling {opts.sampleFractionOnes}")
            # count number of each class value
            counts = df_fp[target].value_counts()
            logging.info(f"Number of sampling values: {counts.to_dict()}")
            # downsample the majority class by taking a fraction of its samples and combining with all of the minority class
            dfX = df_fp[df_fp[target] == 1.0].append(
                df_fp[df_fp[target] == 0.0].sample(
                    int(min(counts[0], counts[1] / opts.sampleFractionOnes)))
            )
            # convert fingerprints to numpy array
            x = np.array(dfX["fp"].to_list(), dtype=settings.ac_fp_numpy_type, copy=settings.numpy_copy_values)
            # convert target values to numpy array
            y = np.array(dfX[target].to_list(), dtype=settings.nn_target_numpy_type, copy=settings.numpy_copy_values)
        else:
            logging.info("Fraction sampling is OFF")
            # count number of each class value
            counts = df_fp[target].value_counts()
            logging.info(f"Number of sampling values: {counts.to_dict()}")
            # convert fingerprints to numpy array
            x = np.array(df_fp["fp"].to_list(), dtype=settings.ac_fp_numpy_type, copy=settings.numpy_copy_values)
            # convert target values to numpy array
            y = np.array(df_fp[target].to_list(), dtype=settings.nn_target_numpy_type, copy=settings.numpy_copy_values)
        # return the numpy arrays of fingerprints and targets
        return x, y



# This function defines a feedforward neural network (FNN) with the given input size, options, and output bias
def build_fnn_network(input_size: int, opts: options.Options, output_bias=None) -> Model:
    # Set the output bias if it is provided
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Define the number of hidden layers based on the input size
    my_hidden_layers = {"2048": 6, "1024": 5, "999": 5, "512": 4, "256": 3}
    if not str(input_size) in my_hidden_layers.keys():
        print(input_size)
        raise ValueError(
            "Wrong input-size. Must be in {2048, 1024, 999, 512, 256}.")
    nhl = int(math.log2(input_size) / 2 - 1)

    # Create a sequential model
    model = Sequential()

    # Add the first hidden layer
    if opts.activationFunction == "relu":
        model.add(Dense(units=int(input_size / 2),
                        input_dim=input_size,
                        activation="relu",
                        kernel_regularizer=regularizers.l2(opts.l2reg),
                        kernel_initializer="he_uniform"))
        model.add(Dropout(opts.dropout))
    elif opts.activationFunction == "selu":
        model.add(Dense(units=int(input_size / 2),
                        input_dim=input_size,
                        activation="selu",
                        kernel_initializer="lecun_normal"))
        model.add(AlphaDropout(opts.dropout))
    else:
        logging.error("Only 'relu' and 'selu' activation is supported")
        sys.exit(-1)

    # Add additional hidden layers
    for i in range(1, nhl):
        factor_units = 2 ** (i + 1)
        factor_dropout = 2 * i
        if opts.activationFunction == "relu":
            model.add(Dense(units=int(input_size / factor_units),
                            activation="relu",
                            kernel_regularizer=regularizers.l2(opts.l2reg),
                            kernel_initializer="he_uniform"))
            model.add(Dropout(opts.dropout / factor_dropout))
        elif opts.activationFunction == "selu":
            model.add(Dense(units=int(input_size / factor_units),
                            activation="selu",
                            kernel_initializer="lecun_normal"))
            model.add(AlphaDropout(opts.dropout / factor_dropout))
        else:
            logging.error("Only 'relu' and 'selu' activation is supported")
            sys.exit(-1)

    # Add the output layer with a sigmoid activation function and the output bias if provided
    model.add(Dense(units=1, activation='sigmoid',
                    bias_initializer=output_bias))
    return model


# This function defines a shallow neural network (SNN) with the given input size, options, and output bias
def build_snn_network(input_size: int, opts: options.Options, output_bias=None) -> Model:
    # Set the output bias if it is provided
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    # Create a sequential model
    model = Sequential()

    # Add the first hidden layer
    model.add(Dense(input_dim=input_size, units=50,
                    activation="selu", kernel_initializer="lecun_normal"))
    model.add(AlphaDropout(opts.dropout))

    # Add 7 hidden layers with 50 units each, a SELU activation function, and a LeCun normal kernel initializer
    for i in range(7):
        model.add(Dense(units=50, activation="selu", kernel_initializer="lecun_normal"))
        # Add an alpha dropout layer with the given dropout rate
        model.add(AlphaDropout(opts.dropout))

    # Add the output layer with 1 unit and a sigmoid activation function
    # If output_bias is not None, set the bias initializer to the specified value
    model.add(Dense(units=1, activation="sigmoid", bias_initializer=output_bias))

    return model


def balanced_accuracy(y_true, y_pred):
    """
    Computes the balanced accuracy metric.

    Args:
        y_true (tf.Tensor): The true values of the labels.
        y_pred (tf.Tensor): The predicted values of the labels.

    Returns:
        tf.Tensor: The balanced accuracy score.
    """
    y_pred = K.round(y_pred)  # Convert continuous predictions to binary class labels
    tn = K.sum((1 - y_true) * (1 - y_pred))
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    tp = K.sum(y_true * y_pred)
    balanced_acc = (tp / (tp + fn) + tn / (tn + fp)) / 2
    return balanced_acc


def define_single_label_model(input_size: int, opts: options.Options, output_bias=None) -> Model:
    """
    Defines and compiles the single-label neural network model.

    Args:
        input_size (int): The size of the input layer.
        opts (options.Options): The options used in the model.
        output_bias (float): The initial bias for the last sigmoid layer of the model.

    Returns:
        tensorflow.keras.Model: The compiled model.
    """
    # Set the loss function according to the option selected
    if opts.lossFunction == "bce":
        loss_function = BinaryCrossentropy()
    elif opts.lossFunction == "mse":
        loss_function = MeanSquaredError()
    else:
        logging.error(f"Your selected loss is not supported: {opts.lossFunction}.")
        sys.exit("Unsupported loss function")

    # Set the optimizer according to the option selected
    if opts.optimizer == 'Adam':
        my_optimizer = optimizers.Adam(learning_rate=opts.learningRate)
    elif opts.optimizer == 'SGD':
        my_optimizer = optimizers.SGD(lr=opts.learningRate, momentum=0.9)
    else:
        logging.error(f"Your selected optimizer is not supported: {opts.optimizer}.")
        sys.exit("Unsupported optimizer")

    # Set the type of neural network according to the option selected
    if opts.fnnType == "FNN":
        model = build_fnn_network(input_size, opts, output_bias)
    elif opts.fnnType == "SNN":
        model = build_snn_network(input_size, opts, output_bias)
    else:
        raise ValueError(f"Option FNN Type is not \"FNN\" or \"SNN\", but {opts.fnnType}.")

    logging.info(f"Network type: {opts.fnnType}")
    model.summary(print_fn=logging.info)

    # Compile the model with the defined options
    model.compile(loss=loss_function,
                  optimizer=my_optimizer,
                  metrics=[metrics.BinaryAccuracy(name="accuracy"),
                           metrics.AUC(),
                           metrics.Precision(),
                           metrics.Recall(),
                           balanced_accuracy])

    return model


def evaluate_model(x_test: np.ndarray, y_test: np.ndarray, file_prefix: str, model: Model,
                   target: str, fold: int) -> pd.DataFrame:
    # Log that we're evaluating the model on the test data
    name = path.basename(file_prefix).replace("_", " ")
    logging.info(f"Evaluating trained model '{name}' on test data")

    # Predict the test set to compute MCC, AUC, ROC curve, etc.
    threshold = 0.5  # TODO: Introduce thresholds different from 0.5!
    y_predict = model.predict(x_test).flatten()
    y_predict_int = (y_predict >= threshold).astype(np.short)
    y_test_int = y_test.astype(np.short)

    # Save the predictions to a CSV file
    (pd
     .DataFrame({
         "y_true": y_test_int,
         "y_predicted": y_predict,
         "y_predicted_int": y_predict_int,
         "target": target,
         "fold": fold})
     .to_csv(path_or_buf=f"{file_prefix}.predicted.testdata.csv")
     )

    # Compute the confusion matrix
    cfm = confusion_matrix(y_true=y_test_int, y_pred=y_predict_int)

    # Compute the metrics that depend on the class label
    precision_recall = classification_report(y_test_int, y_predict_int, output_dict=True)
    prf = pd.DataFrame.from_dict(precision_recall)[['0', '1']]

    # Add balanced accuracy to the computed metrics
    prf.to_csv(path_or_buf=f"{file_prefix}.predicted.testdata.prec_rec_f1.csv")

    # Evaluate the model on the validation set and log the results
    loss, acc, auc_value, precision, recall, balanced_acc = tuple(
        model.evaluate(x=x_test, y=y_test))

    logging.info(f"Loss: {round(loss, 4)}")
    logging.info(f"Accuracy: {round(acc, 4)}")
    logging.info(f"AUC: {round(auc_value, 4)}")
    logging.info(f"Precision: {round(precision, 4)}")
    logging.info(f"Recall: {round(recall, 4)}")
    logging.info(f"Balanced Accuracy: {round(balanced_acc, 4)}")

    # Compute the Matthews Correlation Coefficient (MCC)
    MCC = matthews_corrcoef(y_test_int, y_predict_int)
    logging.info(f"MCC: {round(MCC, 4)}")

    # Generate the AUC-ROC curve data from the validation data
    FPR, TPR, thresholds_keras = roc_curve(y_true=y_test_int, y_score=y_predict,
                                           drop_intermediate=False)
    AUC = auc(FPR, TPR)

    # Save AUC data to CSV file
    pd.DataFrame(list(zip(FPR, TPR, [AUC] * len(FPR), [target] * len(FPR), [fold] * len(FPR))),
                 columns=['fpr', 'tpr', 'auc_value', 'target', 'fold']). \
        to_csv(path_or_buf=f"{file_prefix}.predicted.testdata.aucdata.csv")

    # Generate and save AUC-ROC curve plot
    pl.plot_auc(fpr=FPR, tpr=TPR, target=target, auc_value=AUC,
                filename=f"{file_prefix}.predicted.testdata.aucdata.svg", wandb_logging=False)

    # Return a DataFrame containing the computed metrics
    return pd.DataFrame.from_dict({'p_0': prf['0']['precision'],
                                   'r_0': prf['0']['recall'],
                                   'f1_0': prf['0']['f1-score'],
                                   'p_1': prf['1']['precision'],
                                   'r_1': prf['1']['recall'],
                                   'f1_1': prf['1']['f1-score'],
                                   'loss': loss,
                                   'accuracy': acc,
                                   'balanced_accuracy': balanced_acc,
                                   'MCC': MCC,
                                   'AUC': AUC,
                                   'target': target,
                                   'fold': fold}, orient='index').T


def fit_and_evaluate_model(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                           fold: int, target: str, opts: options.Options) -> pd.DataFrame:
    # Print info about training
    logging.info(f"Training of fold number: {fold}")

    # Define file name prefix for saving models
    model_file_prefix = path.join(
        opts.outputDir, f"{target}_single-labeled_Fold-{fold}")

    # Compute class imbalance
    ids, counts = np.unique(y_train, return_counts=True)
    count_dict = dict(zip(ids, counts))

    # Set initial bias for last sigmoid layer to counter class imbalance
    if count_dict[0] == 0:
        initial_bias = None
        logging.info("No zeroes in training labels. Setting initial_bias to None.")
    else:
        initial_bias = np.log([count_dict[1] / count_dict[0]])
        logging.info(f"Initial bias for last sigmoid layer: {initial_bias[0]}")

    # Define model
    model = define_single_label_model(
        input_size=x_train.shape[1], opts=opts, output_bias=initial_bias)

    # Define checkpoint to save model weights during training
    checkpoint_model_weights_path = f"{model_file_prefix}.model.weights.hdf5"
    callback_list = cb.nn_callback(checkpoint_path=checkpoint_model_weights_path,
                                   opts=opts)

    # Train model
    start = time()
    hist = model.fit(x_train, y_train,
                     callbacks=callback_list,
                     epochs=opts.epochs,
                     batch_size=opts.batchSize,
                     verbose=opts.verbose,
                     validation_data=(x_test, y_test)
                     )
    trainTime = str(round((time() - start) / 60, ndigits=2))
    logging.info(f"Computation time for training the single-label model for {target}: {trainTime} min")

    # Save and plot model history
    pd.DataFrame(hist.history).to_csv(
        path_or_buf=f"{model_file_prefix}.history.csv")
    pl.plot_history(history=hist, file=f"{model_file_prefix}.history.svg")

    # Evaluate model
    callback_model = define_single_label_model(
        input_size=x_train.shape[1], opts=opts)
    performance = evaluate_model(x_test=x_test, y_test=y_test, file_prefix=model_file_prefix, model=callback_model,
                                 target=target, fold=fold)

    return performance



def preprocess_dataframe(df: pd.DataFrame, opts: options.Options) -> Tuple[pd.DataFrame, Set[str]]:
    # Find all the target columns in the dataframe
    targets = set(df.columns) - set(['cid', 'ID', 'id','mol_id', 'smiles', 'fp', 'inchi', 'fpcompressed'])
    targets = list(targets)

    # Create a new dataframe with the expanded fingerprints
    if opts.compressFeatures:
        new_df = pd.DataFrame([{x: y for x, y in enumerate(item)} for item in df['fpcompressed'].values.tolist()], index=df.index)
    else:
        new_df = pd.DataFrame([{x: y for x, y in enumerate(item)} for item in df['fp'].values.tolist()], index=df.index)

    # Join the new dataframe to the original dataframe and drop the old fingerprint column
    df = df.join(new_df).drop(columns=['fp'])

    # Find all the irrelevant columns (i.e. not target columns)
    irrelevant_columns = set(df.columns) - set(targets)

    return df, irrelevant_columns



def get_x_y(df: pd.DataFrame, target: str, train_set: pd.DataFrame, test_set: pd.DataFrame, opts: options.Options):
    train_indices = train_set.index
    test_indices = test_set.index
    if opts.compressFeatures:
        x_train = df.iloc[train_indices, 1:(opts.encFPSize+1)].values
        y_train = df.iloc[train_indices][target].values
        x_test = df.iloc[test_indices, 1:(opts.encFPSize+1)].values
        y_test = df.iloc[test_indices][target].values
    else:
        x_train = df.iloc[train_indices, 1:(opts.fpSize+1)].values
        y_train = df.iloc[train_indices][target].values
        x_test = df.iloc[test_indices, 1:(opts.fpSize+1)].values
        y_test = df.iloc[test_indices][target].values
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, y_train, x_test, y_test




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
    targets = [c for c in df.columns if c in [
        "AR", "ER", "ED", "TR", "GR", "PPARg", "Aromatase"]]
    # targets = [c for c in df.columns if c not in ['cid', 'ID', 'id', 'mol_id', 'smiles', 'fp', 'inchi', 'fpcompressed']]
    # Create an empty DataFrame to collect the performance metrics
    all_fold_results = pd.DataFrame(
        columns=['target', 'fold', 'p_1', 'r_1', 'roc_auc', 'prc_auc', 'MCC'])
    if opts.wabTracking and opts.wabTarget != "":
        # For W&B tracking, we only train one target that's specified as wabTarget "ER".
        # In case it's not there, we use the first one available
        if opts.wabTarget in targets:
            targets = [opts.wabTarget]
        elif opts.wabTarget == "":
            targets = targets
        else:
            logging.error(
                f"The specified wabTarget for Weights & Biases tracking does not exist: {opts.wabTarget}")
            targets = [targets[0]]

    # Collect metrics for each fold and target
    performance_list = []
    if opts.split_type == 'random':
        for target in targets:  # [:1]:
            # target=targets[1] # --> only for testing the code
            x, y = prepare_nn_training_data(df, target, opts)
            if x is None:
                continue

            logging.info(
                f"X training matrix of shape {x.shape} and type {x.dtype}")
            logging.info(
                f"Y training matrix of shape {y.shape} and type {y.dtype}")

            if opts.kFolds == 1:
                # for single 'folds' and when sweeping on W&B, we fix the random state
                if opts.wabTracking and opts.aeWabTracking==False:
                    wandb.init(project=f"FFN_{opts.split_type}", group=f"{target}",
                            name=f"{target}_single_fold")
                if opts.wabTracking and opts.aeWabTracking:
                    wandb.init(project=f"AE_{opts.aeSplitType}_FNN_{opts.split_type}",name=f"{target}_single_fold", config=opts, group=f"{target}",reinit=True)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y,
                                                                        test_size=opts.testSize, random_state=1)
                    logging.info(
                        f"Splitting train/test data with fixed random initializer")
                else:
                    x_train, x_test, y_train, y_test = train_test_split(
                        x, y, stratify=y, test_size=opts.testSize)

                performance = fit_and_evaluate_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                                     fold=0, target=target, opts=opts)
                performance_list.append(performance)
                # all_fold_results = pd.concat([all_fold_results, performance], ignore_index=True)
                # save complete model
                trained_model = define_single_label_model(
                    input_size=len(x[0]), opts=opts)
                # trained_model.load_weights(path.join(opts.outputDir, f"{target}_single-labeled_Fold-0.model.weights.hdf5"))
                trained_model.save_weights(
                    path.join(opts.outputDir, f"{target}_single-labeled_Fold-0.model.weights.hdf5"))
                trained_model.save(filepath=path.join(
                    opts.outputDir, f"{target}_saved_model"))

            elif 1 < opts.kFolds < int(x.shape[0] / 100):
                # do a kfold cross-validation
                kfold_c_validator = StratifiedKFold(
                    n_splits=opts.kFolds, shuffle=True, random_state=42)
                fold_no = 1
                # split the data
                for train, test in kfold_c_validator.split(x, y):
                    # for testing use one of the splits:
                    # kf = kfold_c_validator.split(x, y)
                    # train, test = next(kf)
                    if opts.wabTracking and opts.aeWabTracking==False:
                        wandb.init(project=f"FNN_single_task_{opts.split_type}", group=f"{target}",
                                name=f"{target}-{fold_no}",reinit=True)
                    if opts.wabTracking and opts.aeWabTracking:
                        wandb.init(project=f"AE_{opts.aeSplitType}_FNN_{opts.split_type}",name=f"{target}-{fold_no}", config=opts, group=f"{target}",reinit=True)

                    performance = fit_and_evaluate_model(x_train=x[train], x_test=x[test],
                                                         y_train=y[train], y_test=y[test],
                                                         fold=fold_no, target=target, opts=opts)
                    performance_list.append(performance)
                    # all_fold_results = pd.concat([all_fold_results, performance], ignore_index=True)
                    wandb.finish()
                    fold_no += 1

                    # now next fold
                # select and copy best model - how to define the best model?
                best_fold = (
                    pd
                    .concat(performance_list, ignore_index=True)
                    .sort_values(
                        by=['p_1', 'r_1', 'MCC'],
                        ascending=False,
                        ignore_index=True)['fold'][0]
                )
                best_fold_filename = f"{target}_single-labeled_Fold-{best_fold}.best.model.weights.hdf5"

                # copy checkpoint model weights
                # copy checkpoint model weights
                shutil.copy(
                    src=path.join(opts.outputDir, best_fold_filename),
                    dst=path.join(opts.outputDir, best_fold_filename))
                # shutil.copy(
                # src=path.join(opts.outputDir, f"{target}_single-labeled_Fold-{best_fold}.model.weights.hdf5"),
                # dst=path.join(opts.outputDir, f"{target}_single-labeled_Fold-{best_fold}.best.model.weights.hdf5"))
                # save complete model
                best_model = define_single_label_model(
                    input_size=len(x[0]), opts=opts)
                best_model.save_weights(path.join(opts.outputDir, best_fold_filename))

                # best_model.save_weights(path.join(opts.outputDir,
                #                                   f"{target}_single-labeled_Fold-{best_fold}.best.model.weights.hdf5"))
                # best_model.load_weights(path.join(opts.outputDir,
                #                                   f"{target}_single-labeled_Fold-{best_fold}.model.weights.hdf5"))
                # create output directory and store complete model
                best_model.save(filepath=path.join(
                    opts.outputDir, f"{target}_saved_model"))
                # save complete model
            else:
                logging.info("Your selected number of folds for Cross validation is out of range. "
                             "It must be 1 or smaller than 1 hundredth of the number of samples.")
                sys.exit("Number of folds out of range")
        # Save the DataFrame to a csv file
        all_fold_results.to_csv(
            f'all_folds_{opts.split_type}.csv', index=False)
    # For each individual target train a model
    elif opts.split_type == "scaffold_balanced":
        df, irrelevant_columns = preprocess_dataframe(df, opts)
        for idx, target in enumerate(targets):
            relevant_cols = ["smiles"] + list(irrelevant_columns) + [target]
            df_task = df.loc[:, relevant_cols]
            # Drop rows with missing values in the target column
            df_task.dropna(subset=[target], inplace=True)
            df_task.reset_index(drop=True, inplace=True)
            if opts.kFolds == 1:
                train_set, val_set, test_set = scaffold_split(df_task, sizes=(
                    1-opts.testSize, 0.0, opts.testSize), balanced=False, seed=42)
                x_train, y_train, x_test, y_test = get_x_y(
                    df_task, target, train_set, test_set, opts)
                if opts.wabTracking and not opts.aeWabTracking:
                    wandb.init(project=f"FFN_{opts.split_type}", group=f"{target}",
                            name=f"{target}_single_fold")
                elif opts.wabTracking and opts.aeWabTracking:
                    wandb.init(project=f"AE_{opts.aeSplitType}_FNN_{opts.split_type}",name=f"{target}-{fold_no}", config=opts, group=f"{target}")
        
                performance = fit_and_evaluate_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                                     fold=0, target=target, opts=opts)
                performance_list.append(performance)
                # all_fold_results = pd.concat([all_fold_results, performance], ignore_index=True)

                if opts.compressFeatures:
                    trained_model = define_single_label_model(
                        input_size=opts.encFPSize, opts=opts)
                else:
                    trained_model = define_single_label_model(
                        input_size=opts.fpSize, opts=opts)
                trained_model.save_weights(
                    path.join(opts.outputDir, f"{target}_single-labeled_Fold-0.model.weights.hdf5"))
                trained_model.save(filepath=path.join(
                    opts.outputDir, f"{target}_saved_model"))
            elif opts.kFolds > 1:
                best_fold_dict = dict(p_0=0, r_0=0, f1_0=0, p_1=0, r_1=0, f1_1=0, loss=10, accuracy=0,
                                      balanced_accuracy=0, MCC=0, AUC=0, fold=0)
                for fold_no in range(1, opts.kFolds + 1):
                    print(f"Splitting data with seed {fold_no}")
                    train_set, val_set, test_set = scaffold_split(df_task, sizes=(
                        1-opts.testSize, 0.0, opts.testSize), balanced=True, seed=fold_no)
                    x_train, y_train, x_test, y_test = get_x_y(
                        df_task, target, train_set, test_set, opts)
                    if opts.wabTracking and not opts.aeWabTracking:
                        wandb.init(project=f"FFN_scaffoldAE_{opts.split_type}", group=f"{target}",
                                name=f"{target}-{fold_no}",reinit=True)
                    elif opts.wabTracking and opts.aeWabTracking:
                        wandb.init(project=f"AE_{opts.aeSplitType}_FNN_{opts.split_type}",name=f"{target}-{fold_no}", config=opts, group=f"{target}",reinit=True)
                    performance = fit_and_evaluate_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                                         fold=fold_no, target=target, opts=opts)
                    performance_list.append(performance)

                    if opts.wabTracking:
                        wandb.finish()
                    if opts.compressFeatures:
                        trained_model = define_single_label_model(
                            input_size=opts.encFPSize, opts=opts)
                    else:
                        trained_model = define_single_label_model(
                            input_size=opts.fpSize, opts=opts)
                    trained_model.save_weights(
                        path.join(opts.outputDir, f"{target}_single-labeled_Fold-{fold_no}.scaf.model.weights.hdf5"))
                    trained_model.save(filepath=path.join(
                        opts.outputDir, f"{target}_scaf_saved_model_{fold_no}"))

                    if performance['f1_1'].iloc[0] > best_fold_dict['f1_1'] or performance['accuracy'].iloc[0] > \
                            best_fold_dict["accuracy"]:
                        best_fold_dict['fold_no'] = fold_no

                # Save the best model
                if fold_no == best_fold_dict['fold_no']:
                    if opts.compressFeatures:
                        best_model = define_single_label_model(
                            input_size=opts.encFPSize, opts=opts)
                    else:
                        best_model = define_single_label_model(
                            input_size=opts.fpSize, opts=opts)
                    best_model.save_weights(
                        path.join(opts.outputDir, f"{target}_single-labeled_best_fold-{fold_no}.scaf.model.weights.hdf5"))
                    best_model.save(filepath=path.join(
                        opts.outputDir, f"{target}_scaf_saved_model_best_fold"))
                # Save the DataFrame to a csv file

            else:
                logging.info("Your selected number of folds for Cross validation is out of range. "
                             "It must be 1 or smaller than 1 hundredth of the number of samples.")
                sys.exit("Number of folds out of range")
        (pd
         .concat(performance_list, ignore_index=True)
         .to_csv(path_or_buf=path.join(opts.outputDir, 'single_label_scaf_model.evaluation.csv'))
         )
    else:
        raise Exception(f"Unsupported split type: {opts.split_type}")
