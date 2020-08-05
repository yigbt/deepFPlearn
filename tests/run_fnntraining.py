import pathlib
import logging

import dfpl.options as opt
import dfpl.fingerprint as fp
import dfpl.autoencoder as ac
import dfpl.feedforwardNN as fnn

project_directory = pathlib.Path(__file__).parent.parent.absolute()
test_train_args = opt.TrainOptions(
    inputFile=f"{project_directory}/data/Sun_etal_dataset.csv",
    outputDir=f"{project_directory}/modeltraining/",
    acFile=f"{project_directory}/modeltraining/Sun_etal_encoder.weights.hdf5",
    type='smiles',
    fpType='topological',
    epochs=10,
    fpSize=2048,
    encFPSize=256,
    enableMultiLabel=False,
    testingFraction=0.2,
    kFolds=2,
    verbose=1,
    trainAC=True,
    trainFNN=True
)

def runFNNtraining(opts: opt.TrainOptions) -> None:
    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    logging.info("Adding fingerprint to dataset")
    df = fp.processInParallel(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)

    encoder = None
    if opts.trainAC:
        logging.info("Training autoencoder")
        encoder = ac.trainfullac(df, opts)
        encoder.save_weights(opts.acFile)
    else:
        logging.info("Using trained autoencoder")
        (autoencoder, encoder) = ac.autoencoderModel(input_size=opts.fpSize, encoding_dim=opts.encFPSize)

    df = ac.compressfingerprints(df, encoder)

    # train FNNs with compressed features
    logging.info("Training the FNN using compressed input data.")
    fnn.trainNNmodels(df=df, opts=opts, usecompressed=True)

    # train FNNs with uncompressed features
    logging.info("Training the FNN using UNcompressed input data.")
    fnn.trainNNmodels(df=df, opts=opts, usecompressed=False)

    logging.info("Done")

