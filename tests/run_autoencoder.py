import logging
import pathlib

import dfpl.autoencoder as ac
import dfpl.fingerprint as fp
from dfpl.train import TrainOptions
import dfpl.utils as utils

project_directory = pathlib.Path(__file__).parent.absolute()
test_train_args = TrainOptions(
    inputFile=utils.makePathAbsolute(f"{project_directory}/data/S_dataset.csv"),
    ecModelDir=utils.makePathAbsolute(f"{project_directory}/data"),
    outputDir=utils.makePathAbsolute(f"{project_directory}/output"),
    ecWeightsFile="D_datasetdeterministicrandom.autoencoder.weights.hdf5",
    type="smiles",
    fpType="topological",
    aeEpochs=3,
    fpSize=2048,
    encFPSize=256,
    enableMultiLabel=False,
    testSize=0.2,
    kFolds=2,
    verbose=2,
    trainFNN=False,
    trainAC=True,
)


def runAutoencoder(opts: TrainOptions) -> None:
    """
    Run and test auto-encoder
    """
    logging.basicConfig(
        format="DFPL-{levelname}: {message}", style="{", level=logging.INFO
    )
    logging.info("Adding fingerprint to dataset")
    df = fp.importDataFile(
        opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize
    )
    logging.info("Training autoencoder")
    ac.train_full_ac(df, opts)
    logging.info("Done")


if __name__ == "__main__":
    logging.basicConfig(
        format="DFPL-{levelname}: {message}", style="{", level=logging.INFO
    )
    utils.createDirectory(test_train_args.outputDir)
    runAutoencoder(test_train_args)
