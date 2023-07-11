import logging
import os
import pathlib
import sys

# Add the parent directory of the tests directory to the module search path
tests_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(tests_dir)
sys.path.insert(0, parent_dir)
import dfpl.fingerprint as fp
import dfpl.options as opt
import dfpl.utils as utils
import dfpl.vae as vae

project_directory = pathlib.Path(__file__).parent.absolute()
test_train_args = opt.Options(
    inputFile=utils.makePathAbsolute(f"{project_directory}/data/S_dataset.csv"),
    ecModelDir=utils.makePathAbsolute(f"{project_directory}/output_data/modeltraining"),
    outputDir=utils.makePathAbsolute(f"{project_directory}/output_data"),
    ecWeightsFile="",
    type="smiles",
    fpType="topological",
    epochs=11,
    aeEpochs=6,
    fpSize=2048,
    encFPSize=256,
    enableMultiLabel=False,
    testSize=0.2,
    kFolds=5,
    verbose=2,
    trainFNN=False,
    trainAC=True,
    aeType="variational",
    split_type="scaffold_balanced",
)


def runVae(opts: opt.Options) -> None:
    """
    Run and test auto-encoder
    """
    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    logging.info("Adding fingerprint to dataset")
    df = fp.importDataFile(
        opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize
    )
    logging.info("Training VARIATIONAL autoencoder with scaffold_split")
    vae.train_full_vae(df, opts)
    logging.info("Done")


if __name__ == "__main__":
    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    utils.createDirectory(test_train_args.outputDir)
    runVae(test_train_args)
