import logging
import pathlib

import dfpl.autoencoder as ac
import dfpl.fingerprint as fp
import dfpl.options as opt
import dfpl.single_label_model as fNN
import dfpl.utils as utils

project_directory = pathlib.Path(__file__).parent.absolute()
test_train_args = opt.Options(
    inputFile=utils.makePathAbsolute(f"{project_directory}/data/S_dataset.csv"),
    ecModelDir=utils.makePathAbsolute(
        f"{project_directory}/output/fnnTrainingCompressed/"
    ),
    outputDir=utils.makePathAbsolute(f"{project_directory}/output/fnnTraining"),
    ecWeightsFile="",
    type="smiles",
    fpType="topological",
    epochs=11,
    aeEpochs=3,
    fpSize=2048,
    encFPSize=256,
    testSize=0.2,
    kFolds=1,
    verbose=2,
    trainAC=False,
    trainFNN=True,
)


def run_single_label_training(opts: opt.Options) -> None:
    logging.basicConfig(
        format="DFPL-{levelname}: {message}", style="{", level=logging.INFO
    )
    logging.info("Adding fingerprint to dataset")

    opts.outputDir = utils.makePathAbsolute(
        f"{project_directory}/output/fnnTrainingCompressed"
    )
    utils.createDirectory(opts.outputDir)

    df = fp.importDataFile(
        opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize
    )

    t = opts.ecWeightsFile
    opts.ecWeightsFile = opts.outputDir + t

    if opts.trainAC:
        logging.info("Training autoencoder")
        encoder = ac.train_full_ac(df, opts)
        # encoder.save_weights(opts.acFile)
    else:
        logging.info("Using trained autoencoder")
        (_, encoder) = ac.define_ac_model(opts)

    df = ac.compress_fingerprints(df, encoder)

    # train FNNs with compressed features
    logging.info("Training the FNN using compressed input data.")
    opts.compressFeatures = True

    fNN.train_single_label_models(df=df, opts=opts)

    # train FNNs with uncompressed features
    opts.outputDir = utils.makePathAbsolute(
        f"{project_directory}/output/fnnTrainingUncompressed"
    )
    utils.createDirectory(opts.outputDir)
    logging.info("Training the FNN using un-compressed input data.")
    opts.compressFeatures = False
    fNN.train_single_label_models(df=df, opts=opts)

    logging.info("Done")


if __name__ == "__main__":
    run_single_label_training(test_train_args)
