import logging
import pathlib
from os import path

from tensorflow.keras.models import load_model

import dfpl.autoencoder as ac
import dfpl.fingerprint as fp
import dfpl.options as opt
import dfpl.predictions as p
import dfpl.utils as utils

project_directory = pathlib.Path(__file__).parent.absolute()
test_predict_args = opt.Options(
    inputFile=f"{project_directory}/data/smiles.csv",
    outputDir=f"{project_directory}/preds/",
    ecModelDir=utils.makePathAbsolute(
        f"{project_directory}/data/random_split_autoencoder/encoder_model/"
    ),
    fnnModelDir=f"{project_directory}/output/fnnTrainingCompressed/AR_saved_model",
    fpSize=2048,
    type="smiles",
    fpType="topological",
    compressFeatures=True,
)


def test_predictions(opts: opt.Options):
    opts = test_predict_args

    logging.basicConfig(
        format="DFPL-{levelname}: {message}", style="{", level=logging.INFO
    )
    logging.info(f"Predicting compounds in the input file {opts.inputFile}")

    df = fp.importDataFile(
        opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize
    )

    # use_compressed = False
    if opts.ecModelDir:
        # use_compressed = True
        # load trained model for autoencoder
        (autoencoder, encoder) = ac.define_ac_model(opts, output_bias=None)
        encoder = load_model(opts.ecModelDir)
        # compress the fingerprints using the autoencoder
        df = ac.compress_fingerprints(df, encoder)
    df2 = p.predict_values(df=df, opts=opts)

    names_columns = [c for c in df2.columns if c not in ["fp", "fpcompressed"]]

    output_file = path.join(
        opts.outputDir,
        path.basename(path.splitext(opts.inputFile)[0]) + ".predictions.csv",
    )
    df2[names_columns].to_csv(path_or_buf=output_file)

    logging.info(f"Predictions done.\nResults written to '{output_file}'.")


if __name__ == "__main__":
    logging.basicConfig(
        format="DFPL-{levelname}: {message}", style="{", level=logging.INFO
    )
    utils.createDirectory(test_predict_args.outputDir)
    test_predictions(test_predict_args)
