from __future__ import annotations

import argparse
import dataclasses
from argparse import Namespace
from dataclasses import dataclass

from dfpl.train import TrainPredictCommonOptions, load_compression_options


@dataclass
class PredictOptions(TrainPredictCommonOptions):
    """
    Dataclass for all options necessary for inferring the neural nets.
    Corresponds to `dfpl predict`.
    """
    outputFile: str
    fnnModelDir: str


def parseInputPredict(parser_input_predict: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """

    input_predict_general_args = parser_input_predict.add_argument_group("General Configuration")
    input_predict_files_args = parser_input_predict.add_argument_group("Files")
    input_predict_files_args.add_argument(
        "-i",
        "--inputFile",
        metavar="FILE",
        type=str,
        help="The file containing the data for the prediction in (unquoted) "
             "comma separated CSV format. The column named 'smiles' or 'fp'"
             "contains the field to be predicted. Please adjust the type "
             "that should be predicted (fp or smile) with -t option appropriately."
             "An optional column 'id' is used to assign the outcomes to the"
             "original identifiers. If this column is missing, the results are"
             "numbered in the order of their appearance in the input file."
             "A header is expected and respective column names are used.",
        required=True,
    )
    input_predict_files_args.add_argument(
        "-o",
        "--outputDir",
        metavar="DIR",
        type=str,
        help="Prefix of output directory. It will contain a log file and the file specified with --outputFile.",
        default="example/results_predict/",
    )
    input_predict_files_args.add_argument(
        "--outputFile",
        metavar="FILE",
        type=str,
        help="Output csv file name which will contain one prediction per input line. "
             "Default: prefix of input file name.",
        default="results.csv",  # todo: This doesn't look like it will actually become the prefix of the input file name
    )
    input_predict_general_args.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["fp", "smiles"],
        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
        default="fp",
    )
    input_predict_general_args.add_argument(
        "-k",
        "--fpType",
        type=str,
        choices=["topological", "MACCS"],
        help="The type of fingerprint to be generated/used in input file.",
        default="topological",
    )
    input_predict_files_args.add_argument(
        "--ecModelDir",
        type=str,
        metavar="DIR",
        help="The encoder dir where it is saved (if trainAE=True) or "
             "it is loaded from (if trainAE=False). Provide a full path here.",
        default="",
    )
    input_predict_files_args.add_argument(
        "--ecWeightsFile",
        type=str,
        metavar="STR",
        help="The encoder file where it is loaded from, to compress the fingerprints.",
        default="",
    )
    input_predict_files_args.add_argument(
        "--fnnModelDir",
        type=str,
        metavar="DIR",
        help="The directory where the full model of the fnn is loaded from.",
        default="example/results_train/AR_saved_model",
    )
    input_predict_general_args.add_argument(
        "-c",
        "--compressFeatures",
        action="store_true",
        help="Compresses the fingerprints if encoder dir/file is provided",
        default=False,
    )
    input_predict_general_args.add_argument(
        "--aeType",
        type=str,
        choices=["variational", "deterministic"],
        help="Autoencoder type, variational or deterministic.",
        default="deterministic",
    )


def predict(args: Namespace) -> None:
    """
    Run prediction given specific options
    """
    import logging
    import os
    from os import path

    from keras.saving.save import load_model

    from dfpl import fingerprint as fp, autoencoder as ac, vae as vae
    from dfpl.predictions import predict_values

    from dfpl.utils import makePathAbsolute, createDirectory, createLogger

    predict_opts = PredictOptions(**vars(args))
    opts = dataclasses.replace(
        predict_opts,
        inputFile=makePathAbsolute(predict_opts.inputFile),
        outputDir=makePathAbsolute(predict_opts.outputDir),
        outputFile=makePathAbsolute(
            path.join(predict_opts.outputDir, predict_opts.outputFile)
        ),
        ecModelDir=makePathAbsolute(predict_opts.ecModelDir),
        fnnModelDir=makePathAbsolute(predict_opts.fnnModelDir),
    )
    createDirectory(opts.outputDir)
    createLogger(path.join(opts.outputDir, "predict.log"))
    logging.info(
        f"The following arguments are received or filled with default values:\n{args}"
    )

    # import data from file and create DataFrame
    if "tsv" in opts.inputFile:
        df = fp.importDataFile(
            opts.inputFile, import_function=fp.importDstoxTSV, fp_size=opts.fpSize
        )
    else:
        df = fp.importDataFile(
            opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize
        )

    if opts.compressFeatures:
        # load trained model for autoencoder
        compression_options = load_compression_options()
        if opts.aeType == "deterministic":
            (autoencoder, encoder) = ac.define_ac_model(opts=compression_options)
        if opts.aeType == "variational":
            (autoencoder, encoder) = vae.define_vae_model(opts=compression_options)
        # Load trained model for autoencoder
        if opts.ecWeightsFile == "":
            encoder = load_model(opts.ecModelDir)
        else:
            encoder.load_weights(os.path.join(opts.ecModelDir, opts.ecWeightsFile))
        df = ac.compress_fingerprints(df, encoder)

    # Run predictions on the compressed fingerprints and store the results in a dataframe
    df2 = predict_values(df=df, opts=opts)

    # Extract the column names from the dataframe, excluding the 'fp' and 'fpcompressed' columns
    names_columns = [c for c in df2.columns if c not in ["fp", "fpcompressed"]]

    # Save the predicted values to a CSV file in the output directory
    df2[names_columns].to_csv(path_or_buf=path.join(opts.outputDir, opts.outputFile))

    # Log successful completion of prediction and the file path where the results were saved
    logging.info(
        f"Prediction successful. Results written to '{path.join(opts.outputDir, opts.outputFile)}'"
    )
