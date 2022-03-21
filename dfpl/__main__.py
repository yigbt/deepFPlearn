from argparse import Namespace
import logging
import pathlib
import dataclasses
from os import path

from dfpl.utils import makePathAbsolute, createDirectory
from dfpl import options
from dfpl import fingerprint as fp
from dfpl import autoencoder as ac
from dfpl import feedforwardNN as fNN
from dfpl import predictions

project_directory = pathlib.Path(".").parent.parent.absolute()
opts = options.TrainOptions(
    inputFile=f"{project_directory}/data/S_dataset.pkl",
    outputDir=f"{project_directory}/validation/case_S_ABD_bce/",
    ecWeightsFile=f"{project_directory}/validation/case_00/results_AC_D/ac_D.encoder.hdf5",
    type='smiles',
    fpType='topological',
    epochs=100,
    batchSize=128,
    fpSize=2048,
    encFPSize=256,
    enableMultiLabel=False,
    testingFraction=0.2,
    kFolds=2,
    verbose=2,
    trainAC=False,
    trainFNN=True,
    compressFeatures=True,
    lossFunction="mse",
    optimizer="SGD"
)
logging.basicConfig(level=logging.INFO)

test_predict_args = options.PredictOptions(
    inputFile=f"{project_directory}/data/Sun_etal_dataset.cids.predictionSet.csv",
    outputDir=f"{project_directory}/validation/case_01/results/",
    ecWeightsFile=f"/home/hertelj/git-hertelj/deepFPlearn_CODE/validation/case_00/results_AC_S/ac_S.encoder.hdf5",
    model=f"{project_directory}/validation/case_01/results/AR_compressed-True.full.FNN-.model.hdf5",
    target="AR",
    fpSize=2048,
    type="smiles",
    fpType="topological"
)


def train(opts: options.TrainOptions):
    """
    Run the main training procedure
    :param opts: Options defining the details of the training
    """

    df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)

    # Create output dir if it doesn't exist
    createDirectory(opts.outputDir)

    encoder = None
    if opts.trainAC:
        # train an autoencoder on the full feature matrix
        encoder = ac.train_full_ac(df, opts)

    if opts.compressFeatures:

        if not opts.trainAC:
            # load trained model for autoencoder
            (_, encoder) = ac.define_ac_model(input_size=opts.fpSize, encoding_dim=opts.encFPSize)
            encoder.load_weights(makePathAbsolute(opts.ecWeightsFile))

        # compress the fingerprints using the autoencoder
        df = ac.compress_fingerprints(df, encoder)

    if opts.trainFNN:
        # train single label models
        fNN.train_nn_models(df=df, opts=opts)

    # train multi-label models
    if opts.enableMultiLabel:
        fNN.train_nn_models_multi(df=df, opts=opts)


def predict(opts: options.PredictOptions) -> None:
    """
    Run prediction given specific options
    :param opts: Options defining the details of the prediction
    """
    df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)
    # df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)

    # Create output dir if it doesn't exist
    createDirectory(opts.outputDir)

    use_compressed = False
    if opts.ecWeightsFile:
        logging.info(f"Using fingerprint compression with AC {opts.ecWeightsFile}")
        use_compressed = True
        # load trained model for autoencoder
        (_, encoder) = ac.define_ac_model(input_size=opts.fpSize, encoding_dim=opts.encFPSize)
        encoder.load_weights(opts.ecWeightsFile)
        # compress the fingerprints using the autoencoder
        df = ac.compress_fingerprints(df, encoder)

    # predict
    df2 = predictions.predict_values(df=df,
                                     opts=opts,
                                     use_compressed=use_compressed)

    names_columns = [c for c in df2.columns if c not in ['fp', 'fpcompressed']]

    output_file = path.join(opts.outputDir,
                            path.basename(path.splitext(opts.inputFile)[0]) + ".predictions.csv")
    df2[names_columns].to_csv(path_or_buf=output_file)


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
        if prog_args.method == "train":
            train_opts = options.TrainOptions.fromCmdArgs(prog_args)
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
            predict_opts = options.PredictOptions.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                predict_opts,
                inputFile=makePathAbsolute(predict_opts.inputFile),
                outputDir=makePathAbsolute(predict_opts.outputDir)
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
