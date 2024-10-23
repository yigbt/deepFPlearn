from __future__ import annotations

import argparse
import pathlib
from argparse import Namespace
from dataclasses import dataclass

from dfpl.utils import makePathAbsolute


@dataclass
class TrainPredictCommonOptions:
    """
    Dataclass for all options needed both for training and inferring the neural nets.
    Corresponds to `dfpl train` and `dfpl predict`.
    """
    inputFile: str
    outputDir: str
    ecWeightsFile: str
    ecModelDir: str
    type: str
    fpType: str
    compressFeatures: bool
    aeType: str
    fnnType: str
    fpSize: int


@dataclass
class TrainOptions(TrainPredictCommonOptions):
    """
    Dataclass for all options necessary for training the neural nets.
    Corresponds to `dfpl train`.
    """
    epochs: int
    encFPSize: int
    kFolds: int
    testSize: float
    enableMultiLabel: bool
    verbose: int
    trainAC: bool
    trainFNN: bool
    sampleFractionOnes: float
    sampleDown: bool
    split_type: str
    aeSplitType: str
    aeEpochs: int
    aeBatchSize: int
    aeLearningRate: float
    aeLearningRateDecay: float
    aeActivationFunction: str
    batchSize: int
    optimizer: str
    learningRate: float
    learningRateDecay: float
    lossFunction: str
    activationFunction: str
    l2reg: float
    dropout: float
    threshold: float
    visualizeLatent: bool
    gpu: int
    aeWabTracking: bool
    wabTracking: bool
    wabTarget: str


def parseInputTrain(parser_train: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    # Create argument groups
    input_tain_general_args = parser_train.add_argument_group("Model Configuration")
    input_tain_autoencoder_args = parser_train.add_argument_group("Autoencoder Configuration")
    input_tain_training_args = parser_train.add_argument_group("Training Configuration")
    input_tain_tracking_args = parser_train.add_argument_group("Tracking Configuration")

    input_tain_general_args.add_argument(
        "-i",
        "--inputFile",
        metavar="FILE",
        type=str,
        help="The file containing the data for training in "
             "comma separated CSV format.The first column should be smiles.",
        required=True,
    )
    input_tain_general_args.add_argument(
        "-o",
        "--outputDir",
        metavar="DIR",
        type=str,
        help="Prefix of output file name. Trained model and "
             "respective stats will be returned in this directory.",
        default="example/results_train/",  # changes according to mode
    )

    # TODO CHECK WHAT IS TYPE DOING?
    input_tain_general_args.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["fp", "smiles"],
        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
        default="fp",
    )
    input_tain_general_args.add_argument(
        "-thr",
        "--threshold",
        type=float,
        metavar="FLOAT",
        help="Threshold for binary classification.",
        default=0.5,
    )
    input_tain_general_args.add_argument(
        "-gpu",
        "--gpu",
        metavar="INT",
        type=int,
        help="Select which gpu to use by index. If not available, leave empty",
        default=None,
    )
    input_tain_general_args.add_argument(
        "--fpType",
        type=str,
        # todo: A previous comment in class "Options" listed an additional option "atompairs".
        #       Either add this option here or remove this comment.
        choices=["topological", "MACCS"],
        help="The type of fingerprint to be generated/used in input file. MACCS or topological are available.",
        default="topological",
    )
    input_tain_general_args.add_argument(
        "--fpSize",
        type=int,
        help="Length of the fingerprint that should be generated.",
        default=2048,
    )
    input_tain_general_args.add_argument(
        "--compressFeatures",
        action="store_true",
        help="Compresses the fingerprints. Needs a path of a trained autoencoder or needs the trainAC also set to True.",
        default=False,
    )
    input_tain_general_args.add_argument(
        "--enableMultiLabel",
        action="store_true",
        help="Train multi-label classification model. individual models.",
        default=False,
    )
    # Autoencoder Configuration
    input_tain_autoencoder_args.add_argument(
        "-a",
        "--ecWeightsFile",
        type=str,
        metavar="FILE",
        help="The .hdf5 file of a trained encoder",
        default="",
    )
    input_tain_autoencoder_args.add_argument(
        "--ecModelDir",
        type=str,
        metavar="DIR",
        help="The directory where the full encoder will be saved",
        default="example/results_train/AE_encoder/",
    )
    input_tain_autoencoder_args.add_argument(
        "--aeType",
        type=str,
        choices=["variational", "deterministic"],
        help="Autoencoder type, variational or deterministic.",
        default="deterministic",
    )
    input_tain_autoencoder_args.add_argument(
        "--aeEpochs",
        metavar="INT",
        type=int,
        help="Number of epochs for autoencoder training.",
        default=100,
    )
    input_tain_autoencoder_args.add_argument(
        "--aeBatchSize",
        metavar="INT",
        type=int,
        help="Batch size in autoencoder training.",
        default=512,
    )
    input_tain_autoencoder_args.add_argument(
        "--aeActivationFunction",
        type=str,
        choices=["relu", "selu"],
        help="The activation function of the autoencoder.",
        default="relu",
    )
    input_tain_autoencoder_args.add_argument(
        "--aeLearningRate",
        metavar="FLOAT",
        type=float,
        help="Learning rate for autoencoder training.",
        default=0.001,
    )
    input_tain_autoencoder_args.add_argument(
        "--aeLearningRateDecay",
        metavar="FLOAT",
        type=float,
        help="Learning rate decay for autoencoder training.",
        default=0.96,
    )
    input_tain_autoencoder_args.add_argument(
        "--aeSplitType",
        type=str,
        choices=["scaffold_balanced", "random", "molecular_weight"],
        help="Set how the data is split for the autoencoder",
        default="random",
    )
    input_tain_autoencoder_args.add_argument(
        "-d",
        "--encFPSize",
        metavar="INT",
        type=int,
        help="Size of encoded fingerprint (z-layer of autoencoder).",
        default=256,
    )
    # only if autoencoder is trained or loaded
    input_tain_autoencoder_args.add_argument(
        "--visualizeLatent",
        action="store_true",
        help="UMAP the latent space for exploration",
        default=False,
    )
    # Training Configuration
    input_tain_training_args.add_argument(
        "--split_type",
        type=str,
        choices=["scaffold_balanced", "random", "molecular_weight"],
        help="Set how the data is split for the feedforward neural network",
        default="random",
    )
    input_tain_training_args.add_argument(
        "--testSize",
        metavar="FLOAT",
        type=float,
        help="Fraction[0,1] of the dataset that should be used for testing",
        default=0.2,
    )
    input_tain_autoencoder_args.add_argument(
        "--fnnType",
        type=str,
        choices=["FNN", "SNN"],
        help="The type of the feedforward neural network.",
        default="FNN",
    )
    input_tain_training_args.add_argument(
        "-K",
        "--kFolds",
        metavar="INT",
        type=int,
        help="Number of folds for cross-validation.",
        default=1,
    )
    input_tain_training_args.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        help="Verbosity level. O: No additional output, "
             + "1: Some additional output, 2: full additional output",
        default=2,
    )
    input_tain_training_args.add_argument(
        "--trainAC",
        action="store_true",
        help="Trains the autoencoder.",
        default=False,
    )
    input_tain_training_args.add_argument(
        "--trainFNN",
        action="store_false",
        # todo: This argument is confusing.
        #       Users would expect this flag to be called "--no-trainFNN" or something similar.
        #       Proposal: Rename the flag to "--no-trainFNN",
        #                 but use the parameter dest="trainFNN", so that it
        #                 still appears as "trainFNN" attribute in the resulting arg Namespace
        #                 (set to False, if --no-trainFNN is provided).
        help="Deactivates the FNN training.",
        default=True,
    )
    input_tain_training_args.add_argument(
        "--sampleFractionOnes",
        metavar="FLOAT",
        type=float,
        help="This is the desired fraction 1s/0s.only works if --sampleDown is enabled",
        default=0.5,
    )
    input_tain_training_args.add_argument(
        "--sampleDown",
        metavar="BOOL",
        type=bool,
        help="Down sampling of the 0 valued samples.",
        default=False,
    )
    input_tain_training_args.add_argument(
        "-e",
        "--epochs",
        metavar="INT",
        type=int,
        help="Number of epochs for the FNN training",
        default=100,
    )
    # TODO CHECK IF ALL LOSSES MAKE SENSE HERE
    input_tain_training_args.add_argument(
        "--lossFunction",
        type=str,
        choices=["mse", "bce", "focal"],
        help="Loss function for FNN training. mse - mean squared error, bce - binary cross entropy.",
        default="bce",
    )
    # TODO DO I NEED ALL ARGUMENTS TO BE USER SPECIFIED? WHAT DOES THE USER KNOW ABOUT OPTIMIZERS?
    input_tain_training_args.add_argument(
        "--optimizer",
        type=str,
        choices=["Adam", "SGD"],
        help="Optimizer of the FNN.",
        default="Adam",
    )
    input_tain_training_args.add_argument(
        "--batchSize",
        metavar="INT",
        type=int,
        help="Batch size in FNN training.",
        default=128,
    )
    input_tain_training_args.add_argument(
        "--l2reg",
        metavar="FLOAT",
        type=float,
        help="Value for l2 kernel regularizer.",
        default=0.001,
    )
    input_tain_training_args.add_argument(
        "--dropout",
        metavar="FLOAT",
        type=float,
        help="The fraction of data that is dropped out in each dropout layer.",
        default=0.2,
    )
    input_tain_training_args.add_argument(
        "--learningRate",
        metavar="FLOAT",
        type=float,
        help="Learning rate size in FNN training.",
        default=0.000022,
    )
    input_tain_training_args.add_argument(
        "--learningRateDecay",
        metavar="FLOAT",
        type=float,
        help="Learning rate size in FNN training.",
        default=0.96,
    )
    input_tain_training_args.add_argument(
        "--activationFunction",
        type=str,
        choices=["relu", "selu"],
        help="The activation function of the FNN.",
        default="relu",
    )
    # Tracking Configuration
    # Wand & Biases autoencoder tracking
    input_tain_tracking_args.add_argument(
        "--aeWabTracking",
        metavar="BOOL",
        type=bool,
        help="Track autoencoder performance via Weights & Biases.",
        default=False,
    )
    # Wand & Biases FNN tracking
    input_tain_tracking_args.add_argument(
        "--wabTracking",
        metavar="BOOL",
        type=bool,
        help="Track FNN performance via Weights & Biases",
        default=False,
    )
    # Wand & Biases target used for showing training progress
    input_tain_tracking_args.add_argument(
        "--wabTarget",
        metavar="STRING",
        type=str,
        help="Which endpoint to use for tracking performance via Weights & Biases. Should match the column name.",
        default=None,
    )


def load_compression_options() -> TrainOptions:
    """
    This is a utility function that is needed both by `train` and `predict`.
    It loads options from a JSON file
    that are used to instantiate the autoencoder.
    """
    from dfpl.parse import parse_dfpl

    project_directory = pathlib.Path(__file__).parent.absolute()
    args = parse_dfpl("train",
                      configFile=makePathAbsolute(f"{project_directory}/compression.json"))
    return TrainOptions(**vars(args))


def train(args: Namespace):
    """
    Run the main training procedure
    """
    import dataclasses
    import logging
    import os

    from os import path

    from keras.saving.save import load_model

    from dfpl import fingerprint as fp, autoencoder as ac, vae as vae, single_label_model as sl, feedforwardNN as fNN
    from dfpl.utils import makePathAbsolute, createDirectory, createLogger

    train_opts = TrainOptions(**vars(args))
    opts = dataclasses.replace(
        train_opts,
        inputFile=makePathAbsolute(train_opts.inputFile),
        outputDir=makePathAbsolute(train_opts.outputDir),
    )
    createDirectory(opts.outputDir)
    createLogger(path.join(opts.outputDir, "train.log"))
    logging.info(
        f"The following arguments are received or filled with default values:\n{opts}"
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
    # initialize (auto)encoders to None
    encoder = None
    autoencoder = None
    if opts.trainAC:
        if opts.aeType == "deterministic":
            encoder, train_indices, test_indices = ac.train_full_ac(df, opts)
        elif opts.aeType == "variational":
            encoder, train_indices, test_indices = vae.train_full_vae(df, opts)
        else:
            raise ValueError(f"Unknown autoencoder type: {opts.aeType}")

    # if feature compression is enabled
    if opts.compressFeatures:
        if not opts.trainAC:
            # load default options for autoencoder from config file
            compression_options = load_compression_options()
            if opts.aeType == "variational":
                (autoencoder, encoder) = vae.define_vae_model(opts=compression_options)
            else:
                (autoencoder, encoder) = ac.define_ac_model(opts=compression_options)

            if opts.ecWeightsFile == "":
                encoder = load_model(opts.ecModelDir)
            else:
                autoencoder.load_weights(
                    os.path.join(opts.ecModelDir, opts.ecWeightsFile)
                )
        # compress the fingerprints using the autoencoder
        df = ac.compress_fingerprints(df, encoder)
        if opts.visualizeLatent and opts.trainAC:
            ac.visualize_fingerprints(
                df,
                train_indices=train_indices,
                test_indices=test_indices,
                save_as=f"{opts.ecModelDir}/UMAP_{opts.aeSplitType}.png",
            )
        elif opts.visualizeLatent:
            logging.info(
                "Visualizing latent space is only available if you train the autoencoder. Skipping visualization."
            )

    # train single label models if requested
    if opts.trainFNN and not opts.enableMultiLabel:
        sl.train_single_label_models(df=df, opts=opts)

    # train multi-label models if requested
    if opts.trainFNN and opts.enableMultiLabel:
        fNN.train_nn_models_multi(df=df, opts=opts)
