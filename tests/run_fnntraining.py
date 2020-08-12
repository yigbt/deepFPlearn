import pathlib
import logging

import dfpl.options as opt
import dfpl.fingerprint as fp
import dfpl.autoencoder as ac
import dfpl.feedforwardNN as fNN

project_directory = pathlib.Path(__file__).parent.parent.absolute()
test_train_args = opt.TrainOptions(
    inputFile=f"{project_directory}/data/Sun_etal_dataset.csv",
    outputDir=f"{project_directory}/modeltraining/",
    acFile="Sun_etal_encoder.weights.hdf5",
    type='smiles',
    fpType='topological',
    epochs=11,
    fpSize=2048,
    encFPSize=256,
    enableMultiLabel=False,
    testingFraction=0.2,
    kFolds=2,
    verbose=2,
    trainAC=False,
    trainFNN=True
)


def run_fnn_training(opts: opt.TrainOptions) -> None:
    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    logging.info("Adding fingerprint to dataset")
    df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)

    t = opts.acFile
    opts.acFile = opts.outputDir + t

    if opts.trainAC:
        logging.info("Training autoencoder")
        encoder = ac.train_full_ac(df, opts)
        encoder.save_weights(opts.acFile)
    else:
        logging.info("Using trained autoencoder")
        (_, encoder) = ac.define_ac_model(input_size=opts.fpSize, encoding_dim=opts.encFPSize)

    df = ac.compress_fingerprints(df, encoder)

    # train FNNs with compressed features
    logging.info("Training the FNN using compressed input data.")
    fNN.train_nn_models(df=df, opts=opts, use_compressed=True)

    # train FNNs with uncompressed features
    logging.info("Training the FNN using un-compressed input data.")
    fNN.train_nn_models(df=df, opts=opts, use_compressed=False)

    logging.info("Done")


def run_fnn_training_multi(opts: opt.TrainOptions) -> None:

    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    logging.info("Adding fingerprint to dataset")

    df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)

    t = opts.acFile
    opts.acFile = opts.outputDir + t

    if opts.trainAC:
        logging.info("Training autoencoder")
        encoder = ac.train_full_ac(df, opts)
        encoder.save_weights(opts.acFile)
    else:
        logging.info("Using trained autoencoder")
        (_, encoder) = ac.define_ac_model(input_size=opts.fpSize,
                                          encoding_dim=opts.encFPSize)

    df = ac.compress_fingerprints(df, encoder)

    # train FNNs with compressed features
    logging.info("Training the FNN using compressed input data.")
    fNN.train_nn_models_multi(df=df, opts=opts, use_compressed=True)

    # train FNNs with uncompressed features
    logging.info("Training the FNN using un-compressed input data.")
    fNN.train_nn_models_multi(df=df, opts=opts, use_compressed=False)

    logging.info("Done")
