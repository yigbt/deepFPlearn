import logging
import os
import pathlib
import sys
from os import path

# Add the parent directory of the tests directory to the module search path
tests_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(tests_dir)
sys.path.insert(0, parent_dir)
import dfpl.autoencoder as ac
import dfpl.fingerprint as fp
import dfpl.options as opt
import dfpl.predictions as p
import dfpl.utils as utils

project_directory = pathlib.Path(__file__).parent.absolute()
test_predict_args = opt.Options(
    inputFile=f"{project_directory}/data/smiles.csv",
    outputDir=f"{project_directory}/preds/",
    ecWeightsFile=f"{project_directory}/data/D_datasetdeterministicrandom.autoencoder.weights.hdf5",
    fnnModelDir=f"{project_directory}/output_data/fnnTrainingCompressed/AR_saved_model",
    fpSize=2048,
    type="smiles",
    fpType="topological",
)


def test_predictions(opts: opt.Options):
    opts = test_predict_args

    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    logging.info(f"Predicting compounds in the input file {opts.inputFile}")

    df = fp.importDataFile(
        opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize
    )

    use_compressed = False
    if opts.ecWeightsFile:
        use_compressed = True
        # load trained model for autoencoder
        (autoencoder, encoder) = ac.define_ac_model(opts, output_bias=None)
        autoencoder.load_weights(opts.ecWeightsFile)
        # compress the fingerprints using the autoencoder
        df = ac.compress_fingerprints(df, encoder)
    # model = tensorflow.keras.models.load_model(opts.fnnModelDir, compile=False)
    # model.compile(loss=opts.lossFunction, optimizer=opts.optimizer)
    # predict
    df2 = p.predict_values(df=df, opts=opts)

    names_columns = [c for c in df2.columns if c not in ["fp", "fpcompressed"]]

    output_file = path.join(
        opts.outputDir,
        path.basename(path.splitext(opts.inputFile)[0]) + ".predictions.csv",
    )
    df2[names_columns].to_csv(path_or_buf=output_file)

    logging.info(f"Predictions done.\nResults written to '{output_file}'.")


if __name__ == "__main__":
    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    utils.createDirectory(test_predict_args.outputDir)
    test_predictions(test_predict_args)
