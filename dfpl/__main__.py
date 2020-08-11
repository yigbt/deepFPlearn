from argparse import Namespace
import logging
import pathlib
import dataclasses
from os import path

import options
from utils import makePathAbsolute
import fingerprint as fp
import autoencoder as ac
import feedforwardNN as fNN

project_directory = pathlib.Path(__file__).parent.parent.absolute()
test_train_args = options.TrainOptions(
    inputFile=f"{project_directory}/data/Sun_etal_dataset.csv",
    outputDir=f"{project_directory}/modeltraining",
    acFile="",
    type='smiles',
    fpType='topological',
    epochs=512,
    fpSize=2048,
    encFPSize=256,
    enableMultiLabel=False,
    testingFraction=0.2,
    kFolds=5,
    verbose=1,
    trainAC=False,
    trainFNN=True
)


def train(opts: options.TrainOptions):
    """
    The function defining what happens in the main training procedure
    :param opts:
    """
    # read input and generate fingerprints from smiles
    df = fp.processInParallel(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)
    if opts.trainAC:
        # train an autoencoder on the full feature matrix
        encoder = ac.train_full_ac(df, opts)
        encoder.save_weights(path.join(opts.outputDir, opts.acFile))
    else:
        # load trained model for autoencoder
        (_, encoder) = ac.define_ac_model(input_size=opts.fpSize, encoding_dim=opts.encFPSize)
        encoder.load_weights(path.join(opts.outputDir, opts.acFile))

    # compress the fingerprints using the autoencoder
    df = ac.compress_fingerprints(df, encoder)
    # train FNNs with compressed features
    fNN.train_nn_models(df=df, opts=opts, use_compressed=True)

    # train FNNs with uncompressed features
    fNN.train_nn_models(df=df, opts=opts, use_compressed=False)

    # train multi-label models
    # with compressed features
    fNN.train_nn_models_multi(df=df, opts=opts, use_compressed=True)

    # with uncompressed features
    fNN.train_nn_models_multi(df=df, opts=opts, use_compressed=False)


def predict(opts: options.TrainOptions) -> None:

    print(opts)
    # generate X matrix
    # (xpd, ymatrix) = dfpl.XandYfromInput(csvfilename=args.i, rtype=args.t, fptype=args.k,
    #                                      printfp=True, size=args.s, verbose=args.v, returnY=False)
    # # predict values for provided data and model
    # # ypredictions =
    # dfpl.predictValues(modelfilepath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/
    # 2019-10-16_311681247_1000/model.Aromatase.h5", pdx=xpd)
    # ypredictions = dfpl.predictValues(acmodelfilepath=args.ACmodel, modelfilepath=args.model, pdx=xpd)
    #
    # # write predictions to usr provided .csv file
    # pd.DataFrame.to_csv(ypredictions, args.o)
    return None


def createLogger(filename: str) -> None:
    # get root logger and set its level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(filename)
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


if __name__ == '__main__':
    parser = options.createCommandlineParser()
    prog_args: Namespace = parser.parse_args()

    try:
        if prog_args.method == "train":
            train_opts = options.TrainOptions.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                train_opts,
                inputFile=makePathAbsolute(train_opts.inputFile),
                outputDir=makePathAbsolute(train_opts.outputDir)
            )
            createLogger(path.join(fixed_opts.outputDir, "train.log"))
            logging.info(f"The following arguments are received or filled with default values:\n{prog_args}")
            train(fixed_opts)
            exit(0)
        elif prog_args.method == "predict":
            predict_opts = options.TrainOptions.fromCmdArgs(prog_args)
            fixed_opts = dataclasses.replace(
                predict_opts,
                inputFile=makePathAbsolute(predict_opts.inputFile),
                outputDir=makePathAbsolute(predict_opts.outputDir)
            )
            createLogger(path.join(fixed_opts.outputDir, "predict.log"))
            logging.info(f"The following arguments are received or filled with default values:\n{prog_args}")
            predict(fixed_opts)
            exit(0)
    except AttributeError:
        parser.print_usage()
