import pathlib
import logging
import pandas as pd

import dfpl.options as opt
import dfpl.fingerprint as fp
import dfpl.autoencoder as ac

project_directory = pathlib.Path(__file__).parent.absolute()
test_train_args = opt.Options(
    inputFile=f"{project_directory}/data/S_dataset.csv",
    outputDir=f"{project_directory}/modeltraining",
    ecWeightsFile="Sun_etal_dataset.encoder.hdf5",
    type='smiles',
    fpType='topological',
    epochs=11,
    fpSize=2048,
    encFPSize=256,
    enableMultiLabel=False,
    testingFraction=0.2,
    kFolds=5,
    verbose=2,
    trainFNN=False,
    trainAC=True
)


def runAutoencoder(opts: opt.Options) -> None:
    """
    Run and test auto-encoder
    """
    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    logging.info("Adding fingerprint to dataset")
    df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)
    logging.info("Training autoencoder")
    ac.train_full_ac(df, opts)
    logging.info("Done")


if __name__ == '__main__':
    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    ac.train_full_ac(pd.read_pickle("/home/patrick/Workspace/PycharmProjects/deepFPlearn/modeltraining/df.pkl"), test_train_args)
