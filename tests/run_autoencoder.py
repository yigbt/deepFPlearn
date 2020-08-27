import pathlib
import logging
import pandas as pd

import dfpl.options as opt
import dfpl.fingerprint as fp
import dfpl.autoencoder as ac

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

project_directory = pathlib.Path(__file__).parent.parent.absolute()
test_train_args = opt.TrainOptions(
    inputFile=f"{project_directory}/data/dsstox_20160701.tsv",
    outputDir=f"{project_directory}/modeltraining",
    acFile="ACmodel.hdf5",
    type="smiles",
    fpType="topological",
    epochs=3000,
    fpSize=2048,
    encFPSize=256,
    kFolds=5,
    testingFraction=0.2,
    enableMultiLabel=True,
    verbose=2,
    trainAC=True,
    trainFNN=False,
    compressFeatures=True
)


def runAutoencoder(opts: opt.TrainOptions) -> None:
    """
    Run and test auto-encoder
    """
    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    logging.info("Adding fingerprint to dataset")
    df = fp.importDataFile(opts.inputFile, import_function=fp.importDstoxTSV, fp_size=opts.fpSize)
    logging.info("Training autoencoder")
    ac.train_full_ac(df, opts)
    logging.info("Done")


if __name__ == '__main__':
    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    ac.train_full_ac(pd.read_pickle("/home/patrick/Workspace/PycharmProjects/deepFPlearn/modeltraining/df.pkl"), test_train_args)
