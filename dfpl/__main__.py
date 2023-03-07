import os.path
import sys
import cProfile
sys.path.append("./chemprop_repo")
from chemprop_repo import chemprop
from chemprop_repo.chemprop import args, train

# sys.path.append("./CMPNN")
# from CMPNN.cmpnnchemprop.train import *
# from CMPNN.cmpnnchemprop.utils import create_logger
# from CMPNN.training import cross_validate

import pandas as pd
from argparse import Namespace
import logging
import pathlib
import dataclasses
from os import path
import tensorflow as tf
import math
from tensorflow import keras

from dfpl.utils import makePathAbsolute, createDirectory, createArgsFromJson
from dfpl import options
from dfpl import fingerprint as fp
from dfpl import autoencoder as ac
from dfpl import feedforwardNN as fNN
from dfpl import predictions
from dfpl import single_label_model as sl
from dfpl import rbm as rbm

project_directory = pathlib.Path(".").parent.parent.absolute()
test_train_opts = options.Options(
    inputFile=f'{project_directory}/input_datasets/S_dataset.pkl',
    outputDir=f'{project_directory}/output_data/console_test',
    ecWeightsFile=f'{project_directory}/output_data/case_00/AE_S/ae_S.encoder.hdf5',
    ecModelDir=f'{project_directory}/output_data/case_00/AE_S/saved_model',
    type='smiles',
    fpType='topological',
    epochs=100,
    batchSize=1024,
    fpSize=2048,
    encFPSize=256,
    enableMultiLabel=False,
    testSize=0.2,
    kFolds=2,
    verbose=2,
    trainAC=False,
    trainFNN=True,
    compressFeatures=True,
    activationFunction="selu",
    lossFunction='bce',
    optimizer='Adam',
    fnnType='FNN'
)

test_pred_opts = options.Options(
    inputFile=f"{project_directory}/input_datasets/S_dataset.pkl",
    outputDir=f"{project_directory}/output_data/console_test",
    outputFile=f"{project_directory}/output_data/console_test/S_dataset.predictions_ER.csv",
    ecModelDir=f"{project_directory}/output_data/case_00/AE_S/saved_model",
    fnnModelDir=f"{project_directory}/output_data/console_test/ER_saved_model",
    type="smiles",
    fpType="topological"
)


def traincmpnn(opts: options.GnnOptions):
    logger = create_logger(name='traincmpnn')
    print("Training CMPNN...")
    mean_auc_score, std_auc_score = cross_validate(opts, logger)
    print(f'Results: {mean_auc_score:.5f} +/- {std_auc_score:.5f}')


def traindmpnn(opts: options.GnnOptions):
    ignore_elements = ["py/object", "gnn_type"]
    arguments = createArgsFromJson(opts.configFile, ignore_elements, return_json_object=False)
    opts = chemprop.args.TrainArgs().parse_args(arguments)
    print("Training DMPNN...")
    mean_score, std_score = chemprop.train.cross_validate(args=opts, train_func=chemprop.train.run_training)
    print(f'Results: {mean_score:.5f} +/- {std_score:.5f}')


def predictcmpnn(opts: options.GnnOptions) -> None:
    df = pd.read_csv(opts.test_path)
    # df = df.head(30)
    pred, smiles = make_predictions(opts, df.smiles.tolist())
    df = pd.DataFrame({'smiles': smiles})
    for i in range(len(pred[0])):
        df[f'pred_{i}'] = [item[i] for item in pred]
    df.to_csv(f'{opts.save_dir}/{opts.saving_name}', index=False)


def predictdmpnn(opts: options.GnnOptions, JSON_ARG_PATH) -> None:
    ignore_elements = ["py/object", "gnn_type", "checkpoint_paths", "save_dir", "saving_name"]
    arguments, data = createArgsFromJson(JSON_ARG_PATH, ignore_elements, return_json_object=True)
    arguments.append("--preds_path")
    arguments.append("")
    save_dir = data.get("save_dir")
    name = data.get("saving_name")
    opts = chemprop.args.PredictArgs().parse_args(arguments)
    opts.preds_path = save_dir + "/" + name
    df = pd.read_csv(opts.test_path)
    # df = df.head(10)
    smiles = []
    for index, rows in df.iterrows():
        my_list = [rows.smiles]
        smiles.append(my_list)
    # print(smiles)
    pred = chemprop.train.make_predictions(args=opts, smiles=smiles)


def train(opts: options.Options):
    """
    Run the main training procedure
    :param opts: Options defining the details of the training
    """

    df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)

    # Create output dir if it doesn't exist
    createDirectory(opts.outputDir)  # why? we just created that directory in the function before??
    encoder = None
    rbm_model = None
    if opts.trainAC:
        # train an autoencoder on the full feature matrix
        encoder = ac.train_full_ac(df, opts)
    if opts.trainRBM:
        rbm_model = rbm.train_full_rbm(df, opts)
    if opts.compressFeatures:
        if not opts.useRBM:
            if not opts.trainAC:
                # load trained model for autoencoder
                # encoder = keras.models.load_model(opts.ecModelDir)
                (_, encoder) = ac.define_ac_model(opts=options.Options)
                encoder.load_weights(os.path.join(opts.outputDir, opts.ecWeightsFile))

        else:
            if not opts.trainRBM:
                # load trained model for autoencoder
                rbm_model = rbm.define_rbm_model(opts=options.Options, input_size=opts.fpSize,
                                                 encoding_dim=opts.encFPSize)
                x_run = tf.ones((12, opts.fpSize))
                rbm_model.fit(x_run, x_run, epochs=1)
                rbm_model.load_weights(os.path.join(opts.outputDir, opts.ecWeightsFile))

        # compress the fingerprints using the autoencoder
        if not opts.useRBM:
            df = ac.compress_fingerprints(df, encoder)
            # ac.visualize_fingerprints(df, before_col = "fp", after_col = "fpcompressed",output_dir=opts.outputDir,opts=opts)

        else:
            layer_out = round(math.log2(opts.fpSize / opts.encFPSize))
            df = rbm.compress_fingerprints(df, rbm_model, layer_out - 1)
    if opts.trainFNN:
        # train single label models
        sl.train_single_label_models(df=df, opts=opts)

    # train multi-label models
    if opts.enableMultiLabel:
        fNN.train_nn_models_multi(df=df, opts=opts)


def predict(opts: options.Options) -> None:
    """
    Run prediction given specific options
    :param opts: Options defining the details of the prediction
    """
    df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)

    # Create output dir if it doesn't exist
    createDirectory(opts.outputDir)

    if opts.compressFeatures:
        # load trained model for autoencoder
        encoder = keras.models.load_model(opts.ecModelDir)
        # compress the fingerprints using the autoencoder
        if opts.useRBM:
            df = rbm.compress_fingerprints(df, encoder, layer_num=3)
        else:
            df = ac.compress_fingerprints(df, encoder)

    # predict
    df2 = predictions.predict_values(df=df,
                                     opts=opts)

    names_columns = [c for c in df2.columns if c not in ['fp', 'fpcompressed']]

    df2[names_columns].to_csv(path_or_buf=path.join(opts.outputDir, opts.outputFile))
    logging.info(f"Prediction successful. Results written to '{path.join(opts.outputDir, opts.outputFile)}'")


def createLogger(filename: str) -> None:
    """
    Set up a logger for the main function that also saves to a log file
    """
    # get root logger and set its level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(filename, mode="w")
    fh.setLevel(logging.INFO)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatterFile = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatterConsole = logging.Formatter('%(levelname)-8s %(message)s')
    fh.setFormatter(formatterFile)
    ch.setFormatter(formatterConsole)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def main():
    """
    Main function that runs training/prediction defined by command line arguments
    """
    parser = options.createCommandlineParser()
    prog_args: Namespace = parser.parse_args()
    try:
        if prog_args.method == "convert":
            directory = makePathAbsolute(prog_args.f)
            if path.isdir(directory):
                createLogger(path.join(directory, "convert.log"))
                logging.info(f"Convert all data files in {directory}")
                fp.convert_all(directory)
            else:
                raise ValueError("Input directory is not a directory")
        elif prog_args.method == "traingnn":
            traingnn_opts = options.GnnOptions.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                traingnn_opts,
                # inputFile=makePathAbsolute(traingnn_opts.data_path),
                # outputDir=makePathAbsolute(traingnn_opts.save_dir)
            )

            if traingnn_opts.gnn_type == "cmpnn":
                createDirectory(fixed_opts.save_dir)
                traincmpnn(fixed_opts)
            elif traingnn_opts.gnn_type == "dmpnn":
                traindmpnn(fixed_opts)

        elif prog_args.method == "predictgnn":
            predictgnn_opts = options.GnnOptions.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                predictgnn_opts,
                checkpoint_dir=makePathAbsolute(predictgnn_opts.checkpoint_dir),
                test_path=makePathAbsolute(predictgnn_opts.test_path),
                preds_path=makePathAbsolute(predictgnn_opts.preds_path),
                trainAC=False,
                trainFNN=False
            )

            # createLogger(path.join(fixed_opts.save_dir, "predictgnn.log"))
            # logging.info(f"The following arguments are received or filled with default values:\n{prog_args}")
            if predictgnn_opts.gnn_type == "cmpnn":
                createDirectory(fixed_opts.save_dir)
                predictcmpnn(fixed_opts)
            elif predictgnn_opts.gnn_type == "dmpnn":
                predictdmpnn(fixed_opts, prog_args.configFile)

        elif prog_args.method == "train":
            train_opts = options.Options.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                train_opts,
                inputFile=makePathAbsolute(train_opts.inputFile),
                outputDir=makePathAbsolute(train_opts.outputDir)
            )
            createDirectory(fixed_opts.outputDir)
            createLogger(path.join(fixed_opts.outputDir, "train.log"))
            logging.info(f"The following arguments are received or filled with default values:\n{fixed_opts}")
            train(fixed_opts)
        elif prog_args.method == "predict":
            predict_opts = options.Options.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                predict_opts,
                inputFile=makePathAbsolute(predict_opts.inputFile),
                outputDir=makePathAbsolute(predict_opts.outputDir),
                outputFile=makePathAbsolute(path.join(predict_opts.outputDir, predict_opts.outputFile)),
                ecModelDir=makePathAbsolute(predict_opts.ecModelDir),
                fnnModelDir=makePathAbsolute(predict_opts.fnnModelDir),
                trainAC=False,
                trainFNN=False
            )
            createDirectory(fixed_opts.outputDir)
            createLogger(path.join(fixed_opts.outputDir, "predict.log"))
            logging.info(f"The following arguments are received or filled with default values:\n{prog_args}")
            predict(fixed_opts)
    except AttributeError as e:
        print(e)
        parser.print_usage()


if __name__ == '__main__':
    main()
