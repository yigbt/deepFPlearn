from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import jsonpickle
import torch
from chemprop.args import InterpretArgs, PredictArgs, TrainArgs

from dfpl.utils import parseCmdArgs


@dataclass
class Options:
    """
    Dataclass for all options necessary for training the neural nets
    """

    configFile: str = None
    inputFile: str = ""
    outputDir: str = ""  # changes according to mode
    outputFile: str = ""
    ecWeightsFile: str = ""
    ecModelDir: str = ""
    fnnModelDir: str = ""
    type: str = "smiles"
    fpType: str = "topological"  # also "MACCS", "atompairs"
    epochs: int = 100
    fpSize: int = 2048
    encFPSize: int = 256
    kFolds: int = 1
    testSize: float = 0.2
    enableMultiLabel: bool = False
    verbose: int = 2
    trainAC: bool = False
    trainFNN: bool = True
    compressFeatures: bool = False
    sampleFractionOnes: float = 0.5
    sampleDown: bool = False
    split_type: str = "random"
    aeSplitType: str = "random"
    aeType: str = "deterministic"
    aeEpochs: int = 100
    aeBatchSize: int = 512
    aeLearningRate: float = 0.001
    aeLearningRateDecay: float = 0.96
    aeActivationFunction: str = "selu"
    aeOptimizer: str = "Adam"
    fnnType: str = "FNN"
    batchSize: int = 128
    optimizer: str = "Adam"
    learningRate: float = 0.001
    learningRateDecay: float = 0.96
    lossFunction: str = "bce"
    activationFunction: str = "relu"
    l2reg: float = 0.001
    dropout: float = 0.2
    threshold: float = 0.5
    visualizeLatent: bool = False  # only if autoencoder is trained or loaded
    gpu: int = None
    aeWabTracking: bool = False  # Wand & Biases autoencoder tracking
    wabTracking: bool = False  # Wand & Biases FNN tracking
    wabTarget: str = "AR"  # Wand & Biases target used for showing training progress

    def saveToFile(self, file: str) -> None:
        """
        Saves an instance to a JSON file
        """
        jsonFile = Path(file)
        with jsonFile.open("w") as f:
            f.write(jsonpickle.encode(self))

    @classmethod
    def fromCmdArgs(cls, args: argparse.Namespace) -> "Options":
        return parseCmdArgs(cls, args)


@dataclass
class GnnOptions(TrainArgs):
    """
    Dataclass to hold all options used for training the graph models
    """

    total_epochs: int = 30
    save: bool = True
    configFile: str = ""
    data_path: str = ""
    use_compound_names: bool = False
    save_dir: str = ""
    no_cache: bool = False
    features_scaling: bool = True
    use_input_features: str = ""
    cuda: bool = False
    num_lrs: int = 2
    minimize_score: bool = False
    num_tasks: int = 12
    preds_path: str = ""
    test_path: str = ""
    save_preds: bool = True
    calibration_method: str = ""
    uncertainty_method: str = ""
    calibration_path: str = ""
    evaluation_methods: str = ""
    evaluation_scores_path: str = ""
    wabTracking: bool = False
    split_sizes: List[float] = None

    # save_smiles_splits: bool = False

    @classmethod
    def fromCmdArgs(cls, args: argparse.Namespace, json_config: Optional[dict] = None):
        # Initialize with JSON config if provided
        if json_config:
            opts = cls(**json_config)
        else:
            opts = cls()

        # Update with command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                setattr(opts, key, value)

        return opts


class PredictGnnOptions(PredictArgs):
    """
    Dataclass to hold all options used for training the graph models
    """

    configFile: str = ""
    calibration_atom_descriptors_path: str = None
    calibration_features_path: str = None
    calibration_interval_percentile: float = 95
    calibration_method: Optional[
        Literal[
            "zscaling",
            "tscaling",
            "zelikman_interval",
            "mve_weighting",
            "platt",
            "isotonic",
        ]
    ] = None
    calibration_path: str = None
    calibration_phase_features_path: str = None
    drop_extra_columns: bool = False
    dropout_sampling_size: int = 10
    evaluation_methods: List[str] = None
    evaluation_scores_path: str = None
    # no_features_scaling: bool = True
    individual_ensemble_predictions: bool = False
    preds_path: str = None
    regression_calibrator_metric: Optional[Literal["stdev", "interval"]] = None
    test_path: str = None
    uncertainty_dropout_p: float = 0.1
    uncertainty_method: Optional[
        Literal[
            "mve",
            "ensemble",
            "evidential_epistemic",
            "evidential_aleatoric",
            "evidential_total",
            "classification",
            "dropout",
        ]
    ] = None

    @classmethod
    def fromCmdArgs(cls, args: argparse.Namespace, json_config: Optional[dict] = None):
        # Initialize with JSON config if provided
        if json_config:
            opts = cls(**json_config)
        else:
            opts = cls()

        # Update with command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                setattr(opts, key, value)

        return opts


class InterpretGNNoptions(InterpretArgs):
    """
    Dataclass to hold all options used for training the graph models
    """

    configFile: str = "./example/interpret.json"
    data_path: str = "./example/data/smiles.csv"
    batch_size: int = 500
    c_puct: float = 10.0
    max_atoms: int = 20
    min_atoms: int = 8
    prop_delta: float = 0.5
    property_id: List[int] = None
    rollout: int = 20

    @classmethod
    def fromCmdArgs(cls, args: argparse.Namespace, json_config: Optional[dict] = None):
        # Initialize with JSON config if provided
        if json_config:
            opts = cls(**json_config)
        else:
            opts = cls()

        # Update with command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                setattr(opts, key, value)

        return opts


def createCommandlineParser() -> argparse.ArgumentParser:
    """
    Build the parser for arguments with its two subparsers
    """
    parser = argparse.ArgumentParser(prog="deepFPlearn")
    subparsers = parser.add_subparsers(help="Sub programs of deepFPlearn")

    parser_train_gnn = subparsers.add_parser(
        "traingnn", help="Train new GNN models with your data"
    )
    parser_train_gnn.set_defaults(method="traingnn")
    parseTrainGnn(parser_train_gnn)

    parser_predict_gnn = subparsers.add_parser(
        "predictgnn", help="Predict with your GNN models"
    )
    parser_predict_gnn.set_defaults(method="predictgnn")
    parsePredictGnn(parser_predict_gnn)

    parser_interpret_gnn = subparsers.add_parser(
        "interpretgnn", help="Interpret your GNN models"
    )
    parser_interpret_gnn.set_defaults(method="interpretgnn")
    parseInterpretGnn(parser_interpret_gnn)

    parser_train = subparsers.add_parser(
        "train", help="Train new models with your data"
    )
    parser_train.set_defaults(method="train")
    parseInputTrain(parser_train)

    parser_input_predict = subparsers.add_parser(
        "predict", help="Predict your data with existing models"
    )
    parser_input_predict.set_defaults(method="predict")
    parseInputPredict(parser_input_predict)

    parser_convert = subparsers.add_parser(
        "convert", help="Convert known data files to pickle serialization files"
    )
    parser_convert.set_defaults(method="convert")
    parseInputConvert(parser_convert)
    return parser


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

    # Model Configuration
    input_tain_general_args.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="Input JSON file that contains all information for training/predicting.",
        default="example/train.json",
    )
    input_tain_general_args.add_argument(
        "-i",
        "--inputFile",
        metavar="FILE",
        type=str,
        help="The file containing the data for training in "
        "comma separated CSV format.The first column should be smiles.",
        default="tests/data/smiles.csv",
    )
    input_tain_general_args.add_argument(
        "-o",
        "--outputDir",
        metavar="DIR",
        type=str,
        help="Prefix of output file name. Trained model and "
        "respective stats will be returned in this directory.",
        default="example/results_train/",
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
    input_tain_tracking_args.add_argument(
        "--aeWabTracking",
        metavar="BOOL",
        type=bool,
        help="Track autoencoder performance via Weights & Biases.",
        default=False,
    )
    input_tain_tracking_args.add_argument(
        "--wabTracking",
        metavar="BOOL",
        type=bool,
        help="Track FNN performance via Weights & Biases",
        default=False,
    )
    input_tain_tracking_args.add_argument(
        "--wabTarget",
        metavar="STRING",
        type=str,
        help="Which endpoint to use for tracking performance via Weights & Biases. Should match the column name.",
        default=None,
    )


def parseInputPredict(parser_input_predict: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """

    input_predict_general_args = parser_input_predict.add_argument_group("General Configuration")
    input_predict_files_args = parser_input_predict.add_argument_group("Files")
    input_predict_files_args.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="JSON file that contains all information for training/predicting.",
    )
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
        default="tests/data/smiles.csv",
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
        default="results.csv",
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


def parseTrainGnn(parser_train_gnn: argparse.ArgumentParser) -> None:
    train_gnn_general_args = parser_train_gnn.add_argument_group("General Configuration")
    train_gnn_data_args = parser_train_gnn.add_argument_group("Data Configuration")
    train_gnn_files_args = parser_train_gnn.add_argument_group("Files")
    train_gnn_model_args = parser_train_gnn.add_argument_group("Model arguments")
    train_gnn_training_args = parser_train_gnn.add_argument_group("Training Configuration")
    train_gnn_uncertainty_args = parser_train_gnn.add_argument_group("Uncertainty Configuration")
    train_gnn_uncertainty_args.add_argument(
        "--uncertainty_method",
        type=str,
        choices=[
            "mve",
            "ensemble",
            "evidential_epistemic",
            "evidential_aleatoric",
            "evidential_total",
            "classification",
            "dropout",
            "dirichlet",
        ],
        help="Method to use for uncertainty estimation",
        default="none",
    )
    # Uncertainty arguments
    train_gnn_uncertainty_args.add_argument(
        "--calibration_method",
        type=str,
        choices=[
            "zscaling",
            "tscaling",
            "zelikman_interval",
            "mve_weighting",
            "platt",
            "isotonic",
        ],
        help="Method to use for calibration",
        default="none",
    )
    train_gnn_uncertainty_args.add_argument(
        "--calibration_path",
        type=str,
        metavar="FILE",
        help="Path to file with calibration data",
    )

    # General arguments
    train_gnn_general_args.add_argument(
        "--split_key_molecule",
        type=int,
        help="The index of the key molecule used for splitting",
    )
    train_gnn_general_args.add_argument("--pytorch_seed", type=int, help="Seed for pytorch")
    train_gnn_general_args.add_argument(
        "--cache_cutoff",
        type=float,
        help="Maximum number of molecules in dataset to allow caching.",
    )
    train_gnn_general_args.add_argument(
        "--save_preds", help="Saves test split predictions during training", type=bool
    )
    train_gnn_general_args.add_argument("--wabTracking", action="store_true", default=False)
    train_gnn_general_args.add_argument(
        "--cuda", action="store_true", default=False, help="Turn on cuda"
    )
    train_gnn_general_args.add_argument(
        "--save_smiles_splits",
        action="store_true",
        default=False,
        help="Save smiles for each train/val/test splits",
    )
    train_gnn_general_args.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether to skip training and only test the model",
    )
    train_gnn_general_args.add_argument(
        "--gpu",
        type=int,
        choices=list(range(torch.cuda.device_count())),
        help="Which GPU to use",
    )
    train_gnn_general_args.add_argument("--save", type=bool)
    train_gnn_general_args.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Skip non-essential print statements",
    )
    train_gnn_general_args.add_argument(
        "--log_frequency",
        type=int,
        metavar="INT",
        default=10,
        help="The number of batches between each log",
    )
    train_gnn_general_args.add_argument(
        "--no_cache_mol",
        action="store_true",
        default=False,
        help="If raised, Turn off caching rdkit mols",
    )

    # FILES ARGUMENTS
    train_gnn_files_args.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="JSON file that contains all configuration for training/predicting.",
    )
    train_gnn_files_args.add_argument(
        "--save_dir",
        type=str,
        metavar="DIR",
        default="./ckpt/",
        help="Directory where model checkpoints will be saved",
    )
    train_gnn_files_args.add_argument(
        "--checkpoint_dir",
        type=str,
        metavar="DIR",
        default=None,
        help="Directory from which to load model checkpoints"
        "(walks directory and ensembles all models that are found)",
    )
    train_gnn_files_args.add_argument(
        "--checkpoint_path",
        type=str,
        metavar="FILE",
        default=None,
        help="Path to model checkpoint (.pt file)",
    )
    train_gnn_files_args.add_argument(
        "--checkpoint_paths",
        type=str,
        metavar="FILE",
        nargs="*",
        default=None,
        help="Path to model checkpoint (.pt file)",
    )
    train_gnn_files_args.add_argument(
        "--separate_val_path",
        type=str,
        metavar="FILE",
        help="Path to separate val set, optional",
    )
    train_gnn_files_args.add_argument(
        "--separate_val_features_path",
        type=str,
        metavar="FILE",
        nargs="*",
        help="Path to file with features for separate val set",
    )
    train_gnn_files_args.add_argument(
        "--separate_test_path",
        type=str,
        metavar="FILE",
        help="Path to separate test set, optional",
    )
    train_gnn_files_args.add_argument(
        "--separate_test_features_path",
        type=str,
        metavar="FILE",
        nargs="*",
        help="Path to file with features for separate test set",
    )
    train_gnn_files_args.add_argument(
        "--folds_file",
        type=str,
        metavar="FILE",
        default=None,
        help="Optional file of fold labels",
    )
    train_gnn_files_args.add_argument(
        "--val_fold_index",
        type=int,
        metavar="INT",
        default=None,
        help="Which fold to use as val for cross val",
    )
    train_gnn_files_args.add_argument(
        "--test_fold_index",
        type=int,
        metavar="INT",
        default=None,
        help="Which fold to use as test for cross val",
    )
    train_gnn_files_args.add_argument(
        "--crossval_index_dir",
        type=str,
        metavar="DIR",
        help="Directory in which to find cross validation index files",
    )
    train_gnn_files_args.add_argument(
        "--crossval_index_file",
        type=str,
        metavar="FILE",
        help="Indices of files to use as train/val/test"
        "Overrides --num_folds and --seed.",
    )
    train_gnn_files_args.add_argument(
        "--data_weights_path",
        type=str,
        metavar="FILE",
        help="Path where the data weight are saved",
    )
    train_gnn_files_args.add_argument(
        "--features_path",
        type=str,
        metavar="FILE",
        nargs="*",
        help="Path to features to use in FNN (instead of features_generator)",
    )

    train_gnn_files_args.add_argument(
        "--separate_val_phase_features_path", type=str, metavar="FILE"
    )
    train_gnn_files_args.add_argument(
        "--separate_test_phase_features_path", type=str, metavar="FILE"
    )

    train_gnn_files_args.add_argument(
        "--separate_val_atom_descriptors_path", type=str, metavar="FILE"
    )
    train_gnn_files_args.add_argument(
        "--separate_test_atom_descriptors_path", type=str, metavar="FILE"
    )
    # Data related arguments
    train_gnn_data_args.add_argument(
        "--data_path",
        type=str,
        metavar="FILE",
        help="Path to data CSV file",
        default="",
    )
    train_gnn_data_args.add_argument(
        "--use_compound_names",
        action="store_true",
        default=False,
        help="Use when test data file contains compound names in addition to SMILES strings",
    )
    train_gnn_data_args.add_argument(
        "--max_data_size",
        type=int,
        metavar="INT",
        help="Maximum number of data points to load",
    )

    train_gnn_data_args.add_argument(
        "--features_only",
        action="store_true",
        default=False,
        help="Use only the additional features in an FFN, no graph network",
    )
    train_gnn_data_args.add_argument(
        "--dataset_type",
        type=str,
        choices=["classification", "regression", "multiclass"],
        help="Type of dataset, e.g. classification or regression."
        "This determines the loss function used during training.",
        default="regression",
    )  # classification
    train_gnn_data_args.add_argument(
        "--multiclass_num_classes",
        type=int,
        metavar="INT",
        default=3,
        help="Number of classes in multiclass classification",
    )
    train_gnn_data_args.add_argument(
        "--split_type",
        type=str,
        default="random",
        choices=[
            "random",
            "scaffold_balanced",
            "predetermined",
            "crossval",
            "index_predetermined",
        ],
        help="Method of splitting the data into train/val/test",
    )
    train_gnn_data_args.add_argument(
        "--split_sizes",
        type=float,
        metavar="FLOAT",
        nargs=3,
        default=[0.8, 0.2, 0.0],
        help="Split proportions for train/validation/test sets",
    )

    train_gnn_data_args.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed to use when splitting data into train/val/test sets."
        "When `num_folds` > 1, the first fold uses this seed and all"
        "subsequent folds add 1 to the seed.",
    )
    train_gnn_data_args.add_argument(
        "--smiles_columns",
        type=str,
        metavar="STRING",
        help="Name of the smiles columns",
    )

    train_gnn_data_args.add_argument(
        "--target_columns",
        type=str,
        nargs="*",
        metavar="STRING",
        help="Name of the target columns",
    )

    train_gnn_data_args.add_argument(
        "--ignore_columns",
        type=str,
        nargs="*",
        metavar="STRING",
        help="Names of the columns to ignore",
    )
    train_gnn_data_args.add_argument(
        "--num_tasks", type=int, metavar="INT", help="Number of tasks"
    )
    train_gnn_data_args.add_argument(
        "--no_features_scaling",
        action="store_true",
        default=False,
        help="Turn off scaling of features",
    )
    train_gnn_data_args.add_argument(
        "--features_scaling",
        action="store_true",
        default=False,
        help="Turn on scaling of features",
    )
    train_gnn_data_args.add_argument(
        "--use_input_features",
        type=str,
        metavar="STRING",
        help="Turn on scaling of features",
    )

    # Model arguments
    train_gnn_model_args.add_argument(
        "--ensemble_size",
        type=int,
        metavar="INT",
        default=1,
        help="Number of models in ensemble",
    )
    train_gnn_model_args.add_argument(
        "--hidden_size",
        type=int,
        metavar="INT",
        default=300,
        help="Dimensionality of hidden layers in MPN",
    )
    train_gnn_model_args.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Whether to add bias to linear layers",
    )
    train_gnn_model_args.add_argument(
        "--depth",
        type=int,
        metavar="INT",
        default=3,
        help="Number of message passing steps",
    )
    train_gnn_model_args.add_argument(
        "--dropout",
        type=float,
        metavar="FLOAT",
        default=0.0,
        help="Dropout probability",
    )
    train_gnn_model_args.add_argument(
        "--activation",
        type=str,
        default="ReLU",
        choices=["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"],
        help="Activation function",
    )
    train_gnn_model_args.add_argument(
        "--undirected",
        action="store_true",
        default=False,
        help="Undirected edges (always sum the two relevant bond vectors)",
    )
    train_gnn_model_args.add_argument(
        "--ffn_hidden_size",
        type=int,
        metavar="INT",
        default=2,
        help="Hidden dim for higher-capacity FFN (defaults to hidden_size)",
    )
    train_gnn_model_args.add_argument(
        "--ffn_num_layers",
        type=int,
        metavar="INT",
        default=2,
        help="Number of layers in FFN after MPN encoding",
    )
    train_gnn_model_args.add_argument(
        "--atom_messages",
        action="store_true",
        default=False,
        help="Use messages on atoms instead of messages on bonds",
    )

    train_gnn_model_args.add_argument(
        "--num_lrs",
        type=int,
        metavar="INT",
        default=2,
        help="Number of layers in FFN after MPN encoding",
    )
    train_gnn_model_args.add_argument(
        "--checkpoint_frzn", type=str, metavar="STRING", help="Freeze the loaded model"
    )
    # Model arguments
    # model_args.add_argument("--mpn_shared", type=bool, metavar="BOOL")
    train_gnn_model_args.add_argument(
        "--show_individual_scores",
        action="store_true",
        default=True,
        help="Show all scores for individual targets, not just average, at the end",
    )
    train_gnn_model_args.add_argument(
        "--aggregation",
        choices=["mean", "sum", "norm"],
        help="Aggregation scheme for atomic vectors into molecular vectors",
    )
    train_gnn_model_args.add_argument(
        "--aggregation_norm",
        type=int,
        help="For norm aggregation, number by which to divide summed up atomic features",
    )
    # model_args.add_argument("--explicit_h", type=bool, metavar="BOOL",help="A explicit hydrogen")
    train_gnn_model_args.add_argument(
        "--adding_h", type=bool, metavar="BOOL", help="Adding hydrogen"
    )
    # Training arguments
    train_gnn_model_args.add_argument(
        "--class_balance",
        type=bool,
        metavar="BOOL",
        help="Balances the classes across batches",
    )
    train_gnn_model_args.add_argument(
        "--evidential_regularization",
        type=float,
        metavar="FLOAT",
        help="Regularization parameter for evidential loss",
    )
    train_gnn_model_args.add_argument(
        "--overwrite_default_atom_features",
        type=bool,
        metavar="BOOL",
        help="Overwrites default atom features instead of concatenating",
    )
    train_gnn_model_args.add_argument("--no_atom_descriptor_scaling", type=bool, metavar="BOOL")
    train_gnn_model_args.add_argument(
        "--overwrite_default_bond_features",
        type=bool,
        metavar="BOOL",
        help="Overwrites default bond features instead of concatenating",
    )
    train_gnn_model_args.add_argument(
        "--frzn_ffn_layers",
        type=int,
        metavar="INT",
        help="Number of layers in FFN to freeze",
    )
    # model_args.add_argument("--freeze_first_only", type=bool, metavar="BOOL")
    # Training arguments
    train_gnn_training_args.add_argument(
        "--epochs", type=int, metavar="INT", default=30, help="Number of epochs to run"
    )
    train_gnn_training_args.add_argument(
        "--total_epochs",
        type=int,
        metavar="INT",
        default=30,
        help="Number of total epochs to run",
    )
    train_gnn_training_args.add_argument(
        "--batch_size", type=int, metavar="INT", default=50, help="Batch size"
    )
    train_gnn_training_args.add_argument(
        "--warmup_epochs",
        type=int,
        metavar="INT",
        default=2,
        help="Number of epochs during which learning rate increases linearly from"
        "init_lr to max_lr. Afterwards, learning rate decreases exponentially"
        "from max_lr to final_lr.",
    )
    train_gnn_training_args.add_argument(
        "--init_lr",
        type=float,
        metavar="FLOAT",
        default=1e-4,
        help="Initial learning rate",
    )
    train_gnn_training_args.add_argument(
        "--max_lr",
        type=float,
        metavar="FLOAT",
        default=1e-3,
        help="Maximum learning rate",
    )
    train_gnn_training_args.add_argument(
        "--final_lr",
        type=float,
        metavar="FLOAT",
        default=1e-4,
        help="Final learning rate",
    )
    train_gnn_training_args.add_argument(
        "--extra_metrics",
        type=str,
        metavar="STRING",
        nargs="*",
        help="Extra metrics to use",
    )
    train_gnn_training_args.add_argument(
        "--loss_function",
        type=str,
        choices=[
            "mse",
            "bounded_mse",
            "binary_cross_entropy",
            "cross_entropy",
            "mcc",
            "sid",
            "wasserstein",
            "mve",
            "evidential",
            "dirichlet",
        ],
    )
    train_gnn_training_args.add_argument(
        "--grad_clip", type=float, metavar="FLOAT", help="Gradient clipping value"
    )
    train_gnn_training_args.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=[
            "auc",
            "prc-auc",
            "rmse",
            "mae",
            "mse",
            "r2",
            "accuracy",
            "cross_entropy",
        ],
        help="Metric to use during evaluation."
        "Note: Does NOT affect loss function used during training"
        "(loss is determined by the `dataset_type` argument)."
        'Note: Defaults to "auc" for classification and "rmse" for regression.',
    )
    train_gnn_training_args.add_argument(
        "--num_folds",
        type=int,
        metavar="INT",
        default=1,
        help="Number of folds when performing cross validation",
    )


def parsePredictGnn(parser_predict_gnn: argparse.ArgumentParser) -> None:
    predict_gnn_general_args = parser_predict_gnn.add_argument_group("General Configuration")
    predict_gnn_files_args = parser_predict_gnn.add_argument_group("Files")
    predict_gnn_uncertainty_args = parser_predict_gnn.add_argument_group("Uncertainty Configuration")

    predict_gnn_general_args.add_argument(
        "--checkpoint_path",
        type=str,
        metavar="FILE",
        help="Path to model checkpoint (.pt file)",
    )
    predict_gnn_files_args.add_argument(
        "-f",
        "--configFile",
        type=str,
        metavar="FILE",
        help="Path to a .json file containing arguments. CLI arguments will override these.",
    )
    predict_gnn_files_args.add_argument(
        "--test_path",
        type=str,
        help="Path to CSV file for which predictions will be made.",
    )
    predict_gnn_files_args.add_argument(
        "--preds_path",
        type=str,
        help="Predictions output file. CSV or PICKLE file where predictions will be saved.",
    )
    predict_gnn_files_args.add_argument(
        "--calibration_path",
        type=str,
        help="Data file to be used for uncertainty calibration.",
    )
    predict_gnn_files_args.add_argument(
        "--calibration_features_path",
        type=str,
        nargs="+",
        help="Feature data file to be used with the uncertainty calibration dataset.",
    )
    predict_gnn_files_args.add_argument("--calibration_phase_features_path", type=str, help="")
    predict_gnn_files_args.add_argument(
        "--calibration_atom_descriptors_path",
        type=str,
        help="Extra atom descriptors file.",
    )
    predict_gnn_files_args.add_argument(
        "--calibration_bond_descriptors_path",
        type=str,
        help="Extra bond descriptors file. Path to the extra bond descriptors that will be used as bond features to "
        "featurize a given molecule.",
    )

    predict_gnn_general_args.add_argument(
        "--drop_extra_columns",
        action="store_true",
        help="Keep only SMILES and new prediction columns in the test data files.",
    )

    predict_gnn_uncertainty_args.add_argument(
        "--uncertainty_method",
        type=str,
        choices=[
            "mve",
            "ensemble",
            "evidential_epistemic",
            "evidential_aleatoric",
            "evidential_total",
            "classification",
            "dropout",
            "spectra_roundrobin",
            "dirichlet",
        ],
        help="The method of calculating uncertainty.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--calibration_method",
        type=str,
        nargs="+",
        choices=[
            "zscaling",
            "tscaling",
            "zelikman_interval",
            "mve_weighting",
            "platt",
            "isotonic",
        ],
        help="Methods used for calibrating the uncertainty.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--individual_ensemble_predictions",
        action="store_true",
        default=False,
        help="Save individual ensemble predictions.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--evaluation_methods",
        type=str,
        nargs="+",
        help="Methods used for evaluating the uncertainty performance. Only used if the test data provided includes "
        "targets. Available methods are [nll, miscalibration_area, ence, spearman] or any available "
        "classification or multiclass metric.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--evaluation_scores_path",
        type=str,
        help="Location to save the results of uncertainty evaluations.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--uncertainty_dropout_p",
        type=float,
        default=0.1,
        help="The probability to use for Monte Carlo dropout uncertainty estimation.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--dropout_sampling_size",
        type=int,
        default=10,
        help="The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout "
        "used during training.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--calibration_interval_percentile",
        type=float,
        default=95,
        help="Percentile used in calibration methods. Must be in the range (1,100).",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--regression_calibrator_metric",
        type=str,
        choices=["stdev", "interval"],
        help="Regression calibrator output metric. Regression calibrators can output either a stdev or an inverval.",
    )


def parseInterpretGnn(parser_interpret_gnn: argparse.ArgumentParser) -> None:
    interpret_gnn_files_args = parser_interpret_gnn.add_argument_group("Files")
    interpret_gnn_interpret_args = parser_interpret_gnn.add_argument_group("Interpretation Configuration")
    interpret_gnn_files_args.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="Input JSON file that contains all information for interpretation.",
    )
    interpret_gnn_files_args.add_argument(
        "--preds_path",
        type=str,
        metavar="FILE",
        help="Path to CSV file where predictions will be saved",
        default="",
    )
    interpret_gnn_files_args.add_argument(
        "--checkpoint_dir",
        type=str,
        metavar="DIR",
        help="Directory from which to load model checkpoints"
        "(walks directory and ensembles all models that are found)",
        default="./ckpt",
    )
    interpret_gnn_files_args.add_argument(
        "--checkpoint_path",
        type=str,
        metavar="DIR",
        help="Path to model checkpoint (.pt file)",
    )
    interpret_gnn_files_args.add_argument(
        "--data_path",
        type=str,
        metavar="FILE",
        help="Path to CSV file  for which predictions will be made",
    )
    interpret_gnn_interpret_args.add_argument(
        "--max_atoms",
        type=int,
        metavar="INT",
        help="Maximum number of atoms to use for interpretation",
    )

    interpret_gnn_interpret_args.add_argument(
        "--min_atoms",
        type=int,
        metavar="INT",
        help="Minimum number of atoms to use for interpretation",
    )

    interpret_gnn_interpret_args.add_argument(
        "--prop_delta",
        type=float,
        metavar="FLOAT",
        help="The minimum change in the property of interest that is considered significant",
    )
    interpret_gnn_interpret_args.add_argument(
        "--property_id",
        type=int,
        metavar="INT",
        help="The index of the property of interest",
    )
    # write the argument for rollouts
    interpret_gnn_interpret_args.add_argument(
        "--rollout",
        type=int,
        metavar="INT",
        help="The number of rollouts to use for interpretation",
    )


def parseInputConvert(parser_convert: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    parser_convert.add_argument(
        "-f",
        metavar="FILE",
        type=str,
        help="Input directory where your CSV/TSV files are stored.",
        required=True,
        default="",
    )
