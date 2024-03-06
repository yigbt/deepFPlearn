from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List

import jsonpickle
import torch
from chemprop.args import TrainArgs, PredictArgs, InterpretArgs

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
    configFile: str = "./example/traingnn.json"
    data_path: str = "./example/data/tox21.csv"
    use_compound_names: bool = False
    save_dir: str = ""
    no_cache: bool = False
    features_scaling: bool = True
    use_input_features: str = ""
    cuda: bool = False
    num_lrs: int = 2
    minimize_score: bool = False
    num_tasks: int = 12
    preds_path: str = "./tox21dmpnn.csv"
    test_path: str = ""
    save_preds: bool = True
    calibration_method: str = "none"
    uncertainty_method: str = "none"
    calibration_path: str = ""
    evaluation_methods: str = "none"
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

    configFile: str = "./example/predictgnn.json"
    calibration_atom_descriptors_path: str = None
    calibration_features_path: str = None
    calibration_interval_percentile: float = 95
    calibration_method: Literal[
        "zscaling",
        "tscaling",
        "zelikman_interval",
        "mve_weighting",
        "platt",
        "isotonic",
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
    regression_calibrator_metric: Literal["stdev", "interval"] = None
    test_path: str = None
    uncertainty_dropout_p: float = 0.1
    uncertainty_method: Literal[
        "mve",
        "ensemble",
        "evidential_epistemic",
        "evidential_aleatoric",
        "evidential_total",
        "classification",
        "dropout",
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

    parser_predict = subparsers.add_parser(
        "predict", help="Predict your data with existing models"
    )
    parser_predict.set_defaults(method="predict")
    parseInputPredict(parser_predict)

    parser_convert = subparsers.add_parser(
        "convert", help="Convert known data files to pickle serialization files"
    )
    parser_convert.set_defaults(method="convert")
    parseInputConvert(parser_convert)
    return parser


def parseInputTrain(parser: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    # Create argument groups
    general_args = parser.add_argument_group("Model Configuration")
    autoencoder_args = parser.add_argument_group("Autoencoder Configuration")
    training_args = parser.add_argument_group("Training Configuration")
    tracking_args = parser.add_argument_group("Tracking Configuration")

    # Model Configuration
    general_args.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="Input JSON file that contains all information for training/predicting.",
        default=argparse.SUPPRESS,
    )
    general_args.add_argument(
        "-i",
        "--inputFile",
        metavar="FILE",
        type=str,
        help="The file containing the data for training in "
        "comma separated CSV format.The first column should be smiles.",
        default=argparse.SUPPRESS,
    )
    general_args.add_argument(
        "-o",
        "--outputDir",
        metavar="DIR",
        type=str,
        help="Prefix of output file name. Trained model and "
        "respective stats will be returned in this directory.",
        default=argparse.SUPPRESS,
    )
    general_args.add_argument(
        "-t",
        "--type",
        metavar="STRING",
        type=str,
        choices=["fp", "smiles"],
        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
        default=argparse.SUPPRESS,
    )
    general_args.add_argument(
        "-thr",
        "--threshold",
        type=float,
        metavar="FLOAT",
        help="Threshold for binary classification.",
        default=argparse.SUPPRESS,
    )
    general_args.add_argument(
        "-gpu",
        "--gpu",
        metavar="INT",
        type=int,
        help="Select which gpu to use. If not available, leave empty.",
        default=argparse.SUPPRESS,
    )
    general_args.add_argument(
        "-k",
        "--fpType",
        metavar="STR",
        type=str,
        choices=["topological", "MACCS"],  # , 'atompairs', 'torsions'],
        help="The type of fingerprint to be generated/used in input file.",
        default=argparse.SUPPRESS,
    )
    general_args.add_argument(
        "-s",
        "--fpSize",
        type=int,
        help="Size of fingerprint that should be generated.",
        default=argparse.SUPPRESS,
    )
    general_args.add_argument(
        "-c",
        "--compressFeatures",
        metavar="BOOL",
        type=bool,
        help="Should the fingerprints be compressed or not. Activates the autoencoder. ",
        default=argparse.SUPPRESS,
    )
    general_args.add_argument(
        "-m",
        "--enableMultiLabel",
        metavar="BOOL",
        type=bool,
        help="Train multi-label classification model in addition to the individual models.",
        default=argparse.SUPPRESS,
    )
    # Autoencoder Configuration
    autoencoder_args.add_argument(
        "-a",
        "--ecWeightsFile",
        type=str,
        metavar="FILE",
        help="The .hdf5 file of a trained encoder",
        default=argparse.SUPPRESS,
    )
    autoencoder_args.add_argument(
        "--ecModelDir",
        type=str,
        metavar="DIR",
        help="The directory where the full model of the encoder will be saved",
        default=argparse.SUPPRESS,
    )
    autoencoder_args.add_argument(
        "--aeType",
        metavar="STRING",
        type=str,
        choices=["variational", "deterministic"],
        help="Autoencoder type, variational or deterministic.",
        default=argparse.SUPPRESS,
    )
    autoencoder_args.add_argument(
        "--aeEpochs",
        metavar="INT",
        type=int,
        help="Number of epochs for autoencoder training.",
        default=argparse.SUPPRESS,
    )
    autoencoder_args.add_argument(
        "--aeBatchSize",
        metavar="INT",
        type=int,
        help="Batch size in autoencoder training.",
        default=argparse.SUPPRESS,
    )
    autoencoder_args.add_argument(
        "--aeActivationFunction",
        metavar="STRING",
        type=str,
        choices=["relu", "selu"],
        help="The activation function for the hidden layers in the autoencoder.",
        default=argparse.SUPPRESS,
    )
    autoencoder_args.add_argument(
        "--aeLearningRate",
        metavar="FLOAT",
        type=float,
        help="Learning rate for autoencoder training.",
        default=argparse.SUPPRESS,
    )
    autoencoder_args.add_argument(
        "--aeLearningRateDecay",
        metavar="FLOAT",
        type=float,
        help="Learning rate decay for autoencoder training.",
        default=argparse.SUPPRESS,
    )
    autoencoder_args.add_argument(
        "--aeSplitType",
        metavar="STRING",
        type=str,
        choices=["scaffold_balanced", "random", "molecular_weight"],
        help="Set how the data is going to be split for the autoencoder",
        default=argparse.SUPPRESS,
    )
    autoencoder_args.add_argument(
        "-d",
        "--encFPSize",
        metavar="INT",
        type=int,
        help="Size of encoded fingerprint (z-layer of autoencoder).",
        default=argparse.SUPPRESS,
    )
    # Training Configuration
    training_args.add_argument(
        "--split_type",
        metavar="STRING",
        type=str,
        choices=["scaffold_balanced", "random", "molecular_weight"],
        help="Set how the data is going to be split for the feedforward neural network",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "-l",
        "--testSize",
        metavar="FLOAT",
        type=float,
        help="Fraction of the dataset that should be used for testing. Value in [0,1].",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "-K",
        "--kFolds",
        metavar="INT",
        type=int,
        help="K that is used for K-fold cross-validation in the training procedure.",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "-v",
        "--verbose",
        metavar="INT",
        type=int,
        choices=[0, 1, 2],
        help="Verbosity level. O: No additional output, "
        + "1: Some additional output, 2: full additional output",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "--trainAC",
        metavar="BOOL",
        type=bool,
        help="Choose to train or not, the autoencoder based on the input file",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "--trainFNN",
        metavar="BOOL",
        type=bool,
        help="Train the feedforward network either with provided weights.",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "--sampleFractionOnes",
        metavar="FLOAT",
        type=float,
        help="This is the fraction of positive target associations (1s) in comparison to the majority class(0s)."
        "only works if --sampleDown is enabled",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "--sampleDown",
        metavar="BOOL",
        type=bool,
        help="Enable automatic down sampling of the 0 valued samples.",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "-e",
        "--epochs",
        metavar="INT",
        type=int,
        help="Number of epochs that should be used for the FNN training",
        default=argparse.SUPPRESS,
    )

    training_args.add_argument(
        "--lossFunction",
        metavar="STRING",
        type=str,
        choices=["mse", "bce", "focal"],
        help="Loss function to use during training. mse - mean squared error, bce - binary cross entropy.",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "--optimizer",
        metavar="STRING",
        type=str,
        choices=["Adam", "SGD"],
        help='Optimizer to use for backpropagation in the FNN. Possible values: "Adam", "SGD"',
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "--batchSize",
        metavar="INT",
        type=int,
        help="Batch size in FNN training.",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "--l2reg",
        metavar="FLOAT",
        type=float,
        help="Value for l2 kernel regularizer.",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "--dropout",
        metavar="FLOAT",
        type=float,
        help="The fraction of data that is dropped out in each dropout layer.",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "--learningRate",
        metavar="FLOAT",
        type=float,
        help="Learning rate size in FNN training.",
        default=argparse.SUPPRESS,
    )
    training_args.add_argument(
        "--activationFunction",
        metavar="STRING",
        type=str,
        choices=["relu", "selu"],
        help="The activation function for hidden layers in the FNN.",
        default=argparse.SUPPRESS,
    )
    # Tracking Configuration
    tracking_args.add_argument(
        "--aeWabTracking",
        metavar="BOOL",
        type=bool,
        help="Track autoencoder performance via Weights & Biases, see https://wandb.ai.",
        default=argparse.SUPPRESS,
    )
    tracking_args.add_argument(
        "--wabTracking",
        metavar="BOOL",
        type=bool,
        help="Track FNN performance via Weights & Biases, see https://wandb.ai.",
        default=argparse.SUPPRESS,
    )
    tracking_args.add_argument(
        "--wabTarget",
        metavar="STRING",
        type=str,
        choices=["AR", "ER", "ED", "GR", "TR", "PPARg", "Aromatase"],
        help="Which target to use for tracking performance via Weights & Biases, see https://wandb.ai.",
        default=argparse.SUPPRESS,
    )


def parseTrainGnn(parser: argparse.ArgumentParser) -> None:
    general_args = parser.add_argument_group("General Configuration")
    data_args = parser.add_argument_group("Data Configuration")
    files_args = parser.add_argument_group("Files")
    model_args = parser.add_argument_group("Model arguments")
    training_args = parser.add_argument_group("Training Configuration")
    uncertainty_args = parser.add_argument_group("Uncertainty Configuration")
    uncertainty_args.add_argument(
        "--uncertainty_method",
        type=str,
        metavar="STRING",
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
    uncertainty_args.add_argument(
        "--calibration_method",
        type=str,
        metavar="STRING",
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
    uncertainty_args.add_argument(
        "--calibration_path",
        type=str,
        metavar="FILE",
        help="Path to file with calibration data",
    )

    # General arguments
    general_args.add_argument("--split_key_molecule", type=int)
    general_args.add_argument("--pytorch_seed", type=int)
    general_args.add_argument("--cache_cutoff", type=float)
    general_args.add_argument("--save_preds", type=bool)
    general_args.add_argument(
        "--cuda", action="store_true", default=False, help="Turn on cuda"
    )
    general_args.add_argument(
        "--save_smiles_splits",
        action="store_true",
        default=False,
        help="Save smiles for each train/val/test splits for prediction convenience later",
    )
    general_args.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether to skip training and only test the model",
    )
    general_args.add_argument(
        "--gpu",
        type=int,
        choices=list(range(torch.cuda.device_count())),
        help="Which GPU to use",
    )
    general_args.add_argument("--save", type=bool)
    general_args.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Skip non-essential print statements",
    )
    general_args.add_argument(
        "--log_frequency",
        type=int,
        metavar="INT",
        default=10,
        help="The number of batches between each logging of the training loss",
    )
    general_args.add_argument(
        "--no_cuda", action="store_true", default=True, help="Turn off cuda"
    )
    general_args.add_argument(
        "--no_cache",
        action="store_true",
        default=False,
        help="Turn off caching mol2graph computation",
    )

    # FILES ARGUMENTS
    files_args.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="Input JSON file that contains all information for training/predicting.",
    )
    files_args.add_argument(
        "--config_path",
        type=str,
        metavar="FILE",
        help="Path to a .json file containing arguments. Any arguments present in the config"
        "file will override arguments specified via the command line or by the defaults.",
    )
    files_args.add_argument(
        "--save_dir",
        type=str,
        metavar="DIR",
        default="./ckpt/",
        help="Directory where model checkpoints will be saved",
    )
    files_args.add_argument(
        "--checkpoint_dir",
        type=str,
        metavar="DIR",
        default=None,
        help="Directory from which to load model checkpoints"
        "(walks directory and ensembles all models that are found)",
    )
    files_args.add_argument(
        "--checkpoint_path",
        type=str,
        metavar="FILE",
        default=None,
        help="Path to model checkpoint (.pt file)",
    )
    files_args.add_argument(
        "--checkpoint_paths",
        type=str,
        metavar="FILE",
        nargs="*",
        default=None,
        help="Path to model checkpoint (.pt file)",
    )
    files_args.add_argument(
        "--separate_val_path",
        type=str,
        metavar="FILE",
        help="Path to separate val set, optional",
    )
    files_args.add_argument(
        "--separate_val_features_path",
        type=str,
        metavar="FILE",
        nargs="*",
        help="Path to file with features for separate val set",
    )
    files_args.add_argument(
        "--separate_test_path",
        type=str,
        metavar="FILE",
        help="Path to separate test set, optional",
    )
    files_args.add_argument(
        "--separate_test_features_path",
        type=str,
        metavar="FILE",
        nargs="*",
        help="Path to file with features for separate test set",
    )
    files_args.add_argument(
        "--folds_file",
        type=str,
        metavar="FILE",
        default=None,
        help="Optional file of fold labels",
    )
    files_args.add_argument(
        "--val_fold_index",
        type=int,
        metavar="INT",
        default=None,
        help="Which fold to use as val for cross val",
    )
    files_args.add_argument(
        "--test_fold_index",
        type=int,
        metavar="INT",
        default=None,
        help="Which fold to use as test for cross val",
    )
    files_args.add_argument(
        "--crossval_index_dir",
        type=str,
        metavar="DIR",
        help="Directory in which to find cross validation index files",
    )
    files_args.add_argument(
        "--crossval_index_file",
        type=str,
        metavar="FILE",
        help="Indices of files to use as train/val/test"
        "Overrides --num_folds and --seed.",
    )
    files_args.add_argument(
        "--data_weights_path",
        type=str,
        metavar="FILE",
        help="Path where the data weight are saved",
    )
    files_args.add_argument(
        "--features_path",
        type=str,
        metavar="FILE",
        nargs="*",
        help="Path to features to use in FNN (instead of features_generator)",
    )

    files_args.add_argument(
        "--separate_val_phase_features_path", type=str, metavar="FILE"
    )
    files_args.add_argument(
        "--separate_test_phase_features_path", type=str, metavar="FILE"
    )

    files_args.add_argument(
        "--separate_val_atom_descriptors_path", type=str, metavar="FILE"
    )
    files_args.add_argument(
        "--separate_test_atom_descriptors_path", type=str, metavar="FILE"
    )
    # Data related arguments
    data_args.add_argument(
        "--data_path",
        type=str,
        metavar="FILE",
        help="Path to data CSV file",
        default="",
    )
    data_args.add_argument(
        "--use_compound_names",
        action="store_true",
        default=False,
        help="Use when test data file contains compound names in addition to SMILES strings",
    )
    data_args.add_argument(
        "--max_data_size",
        type=int,
        metavar="INT",
        help="Maximum number of data points to load",
    )

    data_args.add_argument(
        "--features_only",
        action="store_true",
        default=False,
        help="Use only the additional features in an FFN, no graph network",
    )
    data_args.add_argument(
        "--dataset_type",
        type=str,
        metavar="STRING",
        choices=["classification", "regression", "multiclass"],
        help="Type of dataset, e.g. classification or regression."
        "This determines the loss function used during training.",
        default="regression",
    )  # classification
    data_args.add_argument(
        "--multiclass_num_classes",
        type=int,
        metavar="INT",
        default=3,
        help="Number of classes when running multiclass classification",
    )
    data_args.add_argument(
        "--split_type",
        type=str,
        metavar="STRING",
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
    data_args.add_argument(
        "--split_sizes",
        type=float,
        metavar="FLOAT",
        nargs=3,
        default=[0.8, 0.2, 0.0],
        help="Split proportions for train/validation/test sets",
    )

    data_args.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed to use when splitting data into train/val/test sets."
        "When `num_folds` > 1, the first fold uses this seed and all"
        "subsequent folds add 1 to the seed.",
    )
    data_args.add_argument(
        "--smiles_columns",
        type=str,
        metavar="STRING",
        help="Name of the smiles columns",
    )

    data_args.add_argument(
        "--target_columns",
        type=str,
        metavar="STRING",
        help="Name of the target columns",
    )

    data_args.add_argument(
        "--ignore_columns",
        type=str,
        metavar="STRING",
        help="Names of the columns to ignore",
    )
    data_args.add_argument(
        "--num_tasks", type=int, metavar="INT", help="NUmber of tasks"
    )
    data_args.add_argument(
        "--no_features_scaling",
        action="store_true",
        default=False,
        help="Turn off scaling of features",
    )
    data_args.add_argument(
        "--features_scaling",
        action="store_true",
        default=False,
        help="Turn on scaling of features",
    )
    data_args.add_argument(
        "--use_input_features",
        type=str,
        metavar="STRING",
        help="Turn on scaling of features",
    )

    # Model arguments
    model_args.add_argument(
        "--ensemble_size",
        type=int,
        metavar="INT",
        default=1,
        help="Number of models in ensemble",
    )
    model_args.add_argument(
        "--hidden_size",
        type=int,
        metavar="INT",
        default=300,
        help="Dimensionality of hidden layers in MPN",
    )
    model_args.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Whether to add bias to linear layers",
    )
    model_args.add_argument(
        "--depth",
        type=int,
        metavar="INT",
        default=3,
        help="Number of message passing steps",
    )
    model_args.add_argument(
        "--dropout",
        type=float,
        metavar="FLOAT",
        default=0.0,
        help="Dropout probability",
    )
    model_args.add_argument(
        "--activation",
        type=str,
        metavar="STRING",
        default="ReLU",
        choices=["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"],
        help="Activation function",
    )
    model_args.add_argument(
        "--undirected",
        action="store_true",
        default=False,
        help="Undirected edges (always sum the two relevant bond vectors)",
    )
    model_args.add_argument(
        "--ffn_hidden_size",
        type=int,
        metavar="INT",
        default=2,
        help="Hidden dim for higher-capacity FFN (defaults to hidden_size)",
    )
    model_args.add_argument(
        "--ffn_num_layers",
        type=int,
        metavar="INT",
        default=2,
        help="Number of layers in FFN after MPN encoding",
    )
    model_args.add_argument(
        "--atom_messages",
        action="store_true",
        default=False,
        help="Use messages on atoms instead of messages on bonds",
    )

    model_args.add_argument(
        "--num_lrs",
        type=int,
        metavar="INT",
        default=2,
        help="Number of layers in FFN after MPN encoding",
    )
    model_args.add_argument("--checkpoint_frzn", type=str, metavar="STRING")

    # Model arguments
    model_args.add_argument("--mpn_shared", type=bool, metavar="BOOL")
    model_args.add_argument(
        "--show_individual_scores",
        action="store_true",
        default=True,
        help="Show all scores for individual targets, not just average, at the end",
    )
    model_args.add_argument("--aggregation", choices=["mean", "sum", "norm"])
    model_args.add_argument("--aggregation_norm", type=int)
    model_args.add_argument("--explicit_h", type=bool, metavar="BOOL")
    model_args.add_argument("--adding_h", type=bool, metavar="BOOL")
    # Training arguments
    model_args.add_argument("--class_balance", type=bool, metavar="BOOL")
    model_args.add_argument("--evidential_regularization", type=float, metavar="FLOAT")
    model_args.add_argument(
        "--overwrite_default_atom_features", type=bool, metavar="BOOL"
    )
    model_args.add_argument("--no_atom_descriptor_scaling", type=bool, metavar="BOOL")
    model_args.add_argument(
        "--overwrite_default_bond_features", type=bool, metavar="BOOL"
    )
    model_args.add_argument("--frzn_ffn_layers", type=int, metavar="INT")
    model_args.add_argument("--freeze_first_only", type=bool, metavar="BOOL")
    # Training arguments
    training_args.add_argument(
        "--epochs", type=int, metavar="INT", default=30, help="Number of epochs to run"
    )
    training_args.add_argument(
        "--total_epochs",
        type=int,
        metavar="INT",
        default=30,
        help="Number of total epochs to run",
    )
    training_args.add_argument(
        "--batch_size", type=int, metavar="INT", default=50, help="Batch size"
    )
    training_args.add_argument(
        "--warmup_epochs",
        type=int,
        metavar="INT",
        default=2,
        help="Number of epochs during which learning rate increases linearly from"
        "init_lr to max_lr. Afterwards, learning rate decreases exponentially"
        "from max_lr to final_lr.",
    )
    training_args.add_argument(
        "--init_lr",
        type=float,
        metavar="FLOAT",
        default=1e-4,
        help="Initial learning rate",
    )
    training_args.add_argument(
        "--max_lr",
        type=float,
        metavar="FLOAT",
        default=1e-3,
        help="Maximum learning rate",
    )
    training_args.add_argument(
        "--final_lr",
        type=float,
        metavar="FLOAT",
        default=1e-4,
        help="Final learning rate",
    )
    training_args.add_argument(
        "--extra_metrics",
        type=str,
        metavar="STRING",
        nargs="*",
        help="Extra metrics to use",
    )
    training_args.add_argument(
        "--loss_function",
        type=str,
        metavar="STRING",
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
    training_args.add_argument("--grad_clip", type=float)
    training_args.add_argument(
        "--metric",
        type=str,
        metavar="STRING",
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
    training_args.add_argument(
        "--num_folds",
        type=int,
        metavar="INT",
        default=1,
        help="Number of folds when performing cross validation",
    )


def parsePredictGnn(parser: argparse.ArgumentParser) -> None:
    general_args = parser.add_argument_group("General Configuration")
    files_args = parser.add_argument_group("Files")
    uncertainty_args = parser.add_argument_group("Uncertainty Configuration")

    general_args.add_argument(
        "--checkpoint_path",
        type=str,
        metavar="FILE",
        help="Path to model checkpoint (.pt file)"
    )
    # general_args.add_argument(
    #     "--no_features_scaling",
    #     action="store_true",
    #     help="Turn on scaling of features",
    # )
    files_args.add_argument(
        "-f",
        "--configFile",
        type=str,
        metavar="FILE",
        help="Path to a .json file containing arguments. Any arguments present in the config"
        "file will override arguments specified via the command line or by the defaults.",
    )
    files_args.add_argument(
        "--test_path",
        type=str,
        help="Path to CSV file containing testing data for which predictions will be made.",
    )
    files_args.add_argument(
        "--preds_path",
        type=str,
        help="Path to CSV or PICKLE file where predictions will be saved.",
    )
    files_args.add_argument(
        "--calibration_path",
        type=str,
        help="Path to data file to be used for uncertainty calibration.",
    )
    files_args.add_argument(
        "--calibration_features_path",
        type=str,
        nargs="+",
        help="Path to features data to be used with the uncertainty calibration dataset.",
    )
    files_args.add_argument("--calibration_phase_features_path", type=str, help="")
    files_args.add_argument(
        "--calibration_atom_descriptors_path",
        type=str,
        help="Path to the extra atom descriptors.",
    )
    files_args.add_argument(
        "--calibration_bond_descriptors_path",
        type=str,
        help="Path to the extra bond descriptors that will be used as bond features to featurize a given molecule.",
    )

    general_args.add_argument(
        "--drop_extra_columns",
        action="store_true",
        help="Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns.",
    )

    uncertainty_args.add_argument(
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
    uncertainty_args.add_argument(
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
        help="Methods used for calibrating the uncertainty calculated with uncertainty method.",
    )
    uncertainty_args.add_argument("--individual_ensemble_predictions",
        action="store_true",
        default=False,
        help="Whether to save individual ensemble predictions.")
    uncertainty_args.add_argument(
        "--evaluation_methods",
        type=str,
        nargs="+",
        help="The methods used for evaluating the uncertainty performance if the test data provided includes targets. Available methods are [nll, miscalibration_area, ence, spearman] or any available classification or multiclass metric.",
    )
    uncertainty_args.add_argument(
        "--evaluation_scores_path",
        type=str,
        help="Location to save the results of uncertainty evaluations.",
    )
    uncertainty_args.add_argument(
        "--uncertainty_dropout_p",
        type=float,
        default=0.1,
        help="The probability to use for Monte Carlo dropout uncertainty estimation.",
    )
    uncertainty_args.add_argument(
        "--dropout_sampling_size",
        type=int,
        default=10,
        help="The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout used during training.",
    )
    uncertainty_args.add_argument(
        "--calibration_interval_percentile",
        type=float,
        default=95,
        help="Sets the percentile used in the calibration methods. Must be in the range (1,100).",
    )
    uncertainty_args.add_argument(
        "--regression_calibrator_metric",
        type=str,
        choices=["stdev", "interval"],
        help="Regression calibrators can output either a stdev or an inverval.",
    )


def parseInterpretGnn(parser: argparse.ArgumentParser) -> None:
    files_args = parser.add_argument_group("Files")
    interpret_args = parser.add_argument_group("Interpretation Configuration")
    files_args.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="Input JSON file that contains all information for interpretation.",
    )
    files_args.add_argument(
        "--preds_path",
        type=str,
        metavar="FILE",
        help="Path to CSV file where predictions will be saved",
        default="",
    )
    files_args.add_argument(
        "--checkpoint_dir",
        type=str,
        metavar="DIR",
        help="Directory from which to load model checkpoints"
        "(walks directory and ensembles all models that are found)",
        default="./ckpt",
    )
    files_args.add_argument(
        "--checkpoint_path",
        type=str,
        metavar="DIR",
        help="Path to model checkpoint (.pt file)",
    )
    files_args.add_argument(
        "--checkpoint_paths",
        type=str,
        metavar="FILE",
        nargs="*",
        help="Path to model checkpoint (.pt file)",
    )
    files_args.add_argument(
        "--data_path",
        type=str,
        metavar="FILE",
        help="Path to CSV file containing testing data for which predictions will be made",
    )
    interpret_args.add_argument(
        "--max_atoms",
        type=int,
        metavar="INT",
        help="Maximum number of atoms to use for interpretation",
    )

    interpret_args.add_argument(
        "--min_atoms",
        type=int,
        metavar="INT",
        help="Minimum number of atoms to use for interpretation",
    )

    interpret_args.add_argument(
        "--prop_delta",
        type=float,
        metavar="FLOAT",
        help="The minimum change in the property of interest that is considered significant",
    )
    interpret_args.add_argument(
        "--property_id",
        type=int,
        metavar="INT",
        help="The index of the property of interest",
    )
    # write the argument for rollouts
    interpret_args.add_argument(
        "--rollout",
        type=int,
        metavar="INT",
        help="The number of rollouts to use for interpretation",
    )


def parseInputConvert(parser: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    parser.add_argument(
        "-f",
        metavar="FILE",
        type=str,
        help="Input directory where your CSV/TSV files are stored.",
        required=True,
        default="",
    )
