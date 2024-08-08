import dataclasses
import logging
import os.path
import pathlib
from argparse import Namespace
from os import path

import chemprop as cp
import pandas as pd
import wandb
from keras.models import load_model

from dfpl.utils import makePathAbsolute, createDirectory
from dfpl import options
from dfpl import fingerprint as fp
from dfpl import autoencoder as ac
from dfpl import feedforwardNN as fNN
from dfpl import predictions
from dfpl import single_label_model as sl
from dfpl import vae as vae
from dfpl.utils import createArgsFromJson, createDirectory, makePathAbsolute

project_directory = pathlib.Path(".").parent.parent.absolute()
test_train_opts = options.Options(
    inputFile=f'{project_directory}/data/input_datasets/tox24_challenge/tox24_challenge_train.csv',
    outputDir=f'{project_directory}/output/tox24_challenge/',
    ecModelDir=f'{project_directory}/example/models/generic_encoder/',
    type='smiles',
    fpType='topological',
    epochs=10,
    batchSize=47,
    fpSize=2048,
    encFPSize=256,
    enableMultiLabel=False,
    testSize=0.2,
    kFolds=2,
    verbose=2,
    trainAC=False,
    trainFNN=True,
    compressFeatures=True,
    activationFunction="tanh",
    lossFunction='rsme',
    optimizer='Adam',
    fnnType='REG',  # todo: replace useRegressionModel with fnnType variable
    wabTarget='activity_scaled',
    wabTracking=True
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


def traindmpnn(opts: options.GnnOptions):
    """
    Train a D-MPNN model using the given options.
    Args:
    - opts: options.GnnOptions instance containing the details of the training
    Returns:
    - None
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opts.gpu}"
    ignore_elements = ["py/object"]
    # Load options from a JSON file and replace the relevant attributes in `opts`
    arguments = createArgsFromJson(
        opts.configFile, ignore_elements, return_json_object=False
    )
    opts = cp.args.TrainArgs().parse_args(arguments)
    logging.info("Training DMPNN...")
    # Train the model and get the mean and standard deviation of AUC score from cross-validation
    mean_score, std_score = cp.train.cross_validate(
        args=opts, train_func=cp.train.run_training
    )
    logging.info(f"Results: {mean_score:.5f} +/- {std_score:.5f}")


def predictdmpnn(opts: options.GnnOptions, json_arg_path: str) -> None:
    """
    Predict the values using a trained D-MPNN model with the given options.
    Args:
    - opts: options.GnnOptions instance containing the details of the prediction
    - JSON_ARG_PATH: path to a JSON file containing additional arguments for prediction
    Returns:
    - None
    """
    ignore_elements = [
        "py/object",
        "checkpoint_paths",
        "save_dir",
        "saving_name",
    ]
    # Load options and additional arguments from a JSON file
    arguments, data = createArgsFromJson(
        json_arg_path, ignore_elements, return_json_object=True
    )
    arguments.append("--preds_path")
    arguments.append("")
    save_dir = data.get("save_dir")
    name = data.get("saving_name")
    # Replace relevant attributes in `opts` with loaded options
    opts = cp.args.PredictArgs().parse_args(arguments)
    opts.preds_path = save_dir + "/" + name
    df = pd.read_csv(opts.test_path)
    smiles = []
    for index, rows in df.iterrows():
        my_list = [rows.smiles]
        smiles.append(my_list)
    # Make predictions and return the result
    cp.train.make_predictions(args=opts, smiles=smiles)


def train(opts: options.Options):
    """
    Run the main training procedure
    :param opts: Options defining the details of the training
    """

    if opts.wabTracking:
        wandb.init(project=f"dfpl-reg-training-{opts.wabTarget}", entity="dfpl_regression", config=vars(opts))
        # opts = wandb.config

    # df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)
    df = fp.importDataFile(opts.inputFile, import_function=fp.importCSV, fp_size=opts.fpSize)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opts.gpu}"

    # Create output dir if it doesn't exist
    createDirectory(opts.outputDir)  # why? we just created that directory in the function before??

    encoder = None
    if opts.trainAC:
        if opts.aeType == "deterministic":
            encoder, train_indices, test_indices = ac.train_full_ac(df, opts)
        elif opts.aeType == "variational":
            encoder, train_indices, test_indices = vae.train_full_vae(df, opts)
        else:
            raise ValueError(f"Unknown autoencoder type: {opts.aeType}")
    else:
        if opts.aeType == "deterministic":
            (autoencoder, encoder) = ac.define_ac_model(opts=opts)
        elif opts.aeType == "variational":
            (autoencoder, encoder) = vae.define_vae_model(opts=opts)
        else:
            raise ValueError(f"Unknown autoencoder type: {opts.aeType}")

        if opts.ecWeightsFile == "":
            encoder = load_model(opts.ecModelDir)
        else:
            autoencoder.load_weights(
                os.path.join(opts.ecModelDir, opts.ecWeightsFile)
            )

    if opts.compressFeatures:
        # compress the fingerprints using the autoencoder
        df = ac.compress_fingerprints(df, encoder)

    # train single label models if requested
    if opts.trainFNN:
        # train multi-label models if requested
        if opts.enableMultiLabel:
            fNN.train_nn_models_multi(df=df, opts=opts)
        else:
            sl.train_single_label_models(df=df, opts=opts)


def predict(opts: options.Options) -> None:
    """
    Run prediction given specific options
    :param opts: Options defining the details of the prediction
    """
    df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)
    # df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)

    # Create output dir if it doesn't exist
    createDirectory(opts.outputDir)

    if opts.compressFeatures:
        # load trained model for autoencoder
        if opts.aeType == "deterministic":
            (autoencoder, encoder) = ac.define_ac_model(opts=options.Options())
        elif opts.aeType == "variational":
            (autoencoder, encoder) = vae.define_vae_model(opts=options.Options())
        else:
            raise ValueError(f"Unknown autoencoder type: {opts.aeType}")

        # Load trained model for autoencoder
        if opts.ecWeightsFile == "":
            encoder = load_model(opts.ecModelDir)
        else:
            encoder.load_weights(os.path.join(opts.ecModelDir, opts.ecWeightsFile))
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

            traindmpnn(traingnn_opts)

        elif prog_args.method == "predictgnn":
            predictgnn_opts = options.GnnOptions.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                predictgnn_opts,
                test_path=makePathAbsolute(predictgnn_opts.test_path),
                preds_path=makePathAbsolute(predictgnn_opts.preds_path),
            )

            logging.info(
                f"The following arguments are received or filled with default values:\n{prog_args}"
            )

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
            exit(0)
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
            exit(0)
    except AttributeError as e:
        print(e)
        parser.print_usage()


if __name__ == '__main__':
    main()
