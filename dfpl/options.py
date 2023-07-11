from __future__ import annotations

import sys

import argparse
from dataclasses import dataclass
from pathlib import Path

import jsonpickle
import torch
from chemprop.args import TrainArgs

from dfpl.utils import makePathAbsolute


@dataclass
class Options:
    """
    Dataclass for all options necessary for training the neural nets
    """

    configFile: str = "./example/train.json"
    inputFile: str = "/deepFPlearn/CMPNN/data/tox21.csv"
    outputDir: str = "."
    outputFile: str = ""
    ecWeightsFile: str = "AE.encoder.weights.hdf5"
    ecModelDir: str = "AE_encoder"
    fnnModelDir: str = "modeltraining"
    type: str = "smiles"
    fpType: str = "topological"  # also "MACCS", "atompairs"
    epochs: int = 512
    fpSize: int = 2048
    encFPSize: int = 256
    kFolds: int = 0
    testSize: float = 0.2
    enableMultiLabel: bool = False
    verbose: int = 0
    trainAC: bool = True  # if set to False, an AC weight file must be provided!
    trainFNN: bool = True
    compressFeatures: bool = True
    sampleFractionOnes: float = 0.5  # Only used when value is in [0,1]
    sampleDown: bool = False
    split_type: str = "random"
    aeSplitType: str = "random"
    aeType: str = "deterministic"
    aeEpochs: int = 3000
    aeBatchSize: int = 512
    aeLearningRate: float = 0.001
    aeLearningRateDecay: float = 0.01
    aeActivationFunction: str = "relu"
    aeOptimizer: str = "Adam"
    trainRBM: bool = True
    useRBM: bool = True
    fnnType: str = "FNN"
    batchSize: int = 128
    optimizer: str = "Adam"
    learningRate: float = 0.001
    lossFunction: str = "bce"
    activationFunction: str = "relu"
    l2reg: float = 0.001
    dropout: float = 0.2
    threshold: float = 0.5
    gpu: str = ""
    snnDepth = 8
    snnWidth = 50
    aeWabTracking: str = ""  # Wand & Biases autoencoder tracking
    wabTracking: str = ""  # Wand & Biases FNN tracking
    wabTarget: str = "ER"  # Wand & Biases target used for showing training progress

    def saveToFile(self, file: str) -> None:
        """
        Saves an instance to a JSON file
        """
        jsonFile = Path(file)
        with jsonFile.open("w") as f:
            f.write(jsonpickle.encode(self))

    @classmethod
    def fromJson(cls, file: str) -> Options:
        """
        Create an instance from a JSON file
        """
        jsonFile = Path(file)
        if jsonFile.exists() and jsonFile.is_file():
            with jsonFile.open() as f:
                content = f.read()
                return jsonpickle.decode(content)
        raise ValueError("JSON file does not exist or is not readable")

    @classmethod
    def fromCmdArgs(cls, args: argparse.Namespace) -> Options:
        """
        Creates Options instance from cmdline arguments.

        If a training file (JSON) is provided, the values from that file are used.
        However, additional commandline arguments will be preferred. If, e.g., "fpSize" is specified both in the
        JSON file and on the commandline, then the value of the commandline argument will be used.
        """
        result = Options()
        if "configFile" in vars(args).keys():
            jsonFile = Path(makePathAbsolute(args.configFile))
            if jsonFile.exists() and jsonFile.is_file():
                with jsonFile.open() as f:
                    content = f.read()
                    result = jsonpickle.decode(content)
            else:
                raise ValueError("Could not find JSON input file")

        for key, value in vars(args).items():
            # The args dict will contain a "method" key from the subparser.
            # We don't use this.
            if key != "method":
                result.__setattr__(key, value)
        return result


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
    gnn_type: str = "cmpnn"
    preds_path: str = "./tox21dmpnn.csv"
    test_path: str = ""
    save_preds: bool = True


    @classmethod
    def fromCmdArgs(cls, args: argparse.Namespace) -> GnnOptions:
        """
        Creates Options instance from cmdline arguments.

        If a training file (JSON) is provided, the values from that file are used.
        However, additional commandline arguments will be preferred. If, e.g., "fpSize" is specified both in the
        JSON file and on the commandline, then the value of the commandline argument will be used.
        """
        result = GnnOptions()
        if "configFile" in vars(args).keys():
            jsonFile = Path(makePathAbsolute(args.configFile))
            if jsonFile.exists() and jsonFile.is_file():
                with jsonFile.open() as f:
                    content = f.read()
                    result = jsonpickle.decode(content)
            else:
                raise ValueError("Could not find JSON input file")

        return result

    @classmethod
    def fromJson(cls, file: str) -> GnnOptions:
        """
        Create an instance from a JSON file
        """
        jsonFile = Path(file)
        if jsonFile.exists() and jsonFile.is_file():
            with jsonFile.open() as f:
                content = f.read()
                return jsonpickle.decode(content)
        raise ValueError("JSON file does not exist or is not readable")


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
    #     parser = argparse.ArgumentParser(description='Train a DNN to associate chemical fingerprints with a (set of)'
    #                                       'target(s). Trained models are saved to disk including fitted weights and '
    #                                         'can be used in the deepFPlearn-Predict.py tool to make predictions.')
    parser.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="Input JSON file that contains all information for training/predicting.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-i",
        "--inputFile",
        metavar="FILE",
        type=str,
        help="The file containing the data for training in (unquoted) "
        "comma separated CSV format. First column contain the feature string in "
        "form of a fingerprint or a SMILES (see -t option). "
        "The remaining columns contain the outcome(s) (Y matrix). "
        "A header is expected and respective column names are used "
        "to refer to outcome(s) (target(s)).",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        metavar="FILE",
        type=str,
        help="Prefix of output file name. Trained model(s) and "
        "respective stats will be returned in this directory.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--split_type",
        metavar="STR",
        type=str,
        choices=["scaffold_balanced", "random", "molecular_weight"],
        help="Set how the data is going to be split for the feedforward neural network",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aeType",
        metavar="STR",
        type=str,
        choices=["variational", "deterministic"],
        help="Autoencoder type",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aeSplitType",
        metavar="STR",
        type=str,
        choices=["scaffold_balanced", "random", "molecular_weight"],
        help="Set how the data is going to be split for the autoencoder",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-t",
        "--type",
        metavar="STR",
        type=str,
        choices=["fp", "smiles"],
        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-thr",
        "--threshold",
        type=float,
        help="Threshold for binary classification.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-gpu",
        "--gpu",
        metavar="STR",
        type=str,
        help="Select which gpu to use",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-k",
        "--fpType",
        metavar="STR",
        type=str,
        choices=["topological", "MACCS"],  # , 'atompairs', 'torsions'],
        help="The type of fingerprint to be generated/used in input file.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-s",
        "--fpSize",
        type=int,
        help="Size of fingerprint that should be generated.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-c",
        "--compressFeatures",
        metavar="BOOL",
        type=bool,
        help="Compress the fingerprints. This is done either with an existing autoencoder or a new "
        "autoencoder model will be trained using the input compounds (see further options for "
        "details).",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-a",
        "--ecWeightsFile",
        type=str,
        metavar="FILE",
        help="The .hdf5 file of a trained encoder (e.g. from a previous"
        "training run. This avoids a retraining of the autoencoder on the"
        "training data set (provided with -i). NOTE that the input and encoding"
        "dimensions must fit your data and settings. Default: train new autoencoder.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ecModelDir",
        type=str,
        metavar="DIR",
        help="The directory where the full model of the encoder will be saved (if trainAE=True) or "
        "loaded from (if trainAE=False). Provide a full path here.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-d",
        "--encFPSize",
        metavar="INT",
        type=int,
        help="Size of encoded fingerprint (z-layer of autoencoder).",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="INT",
        type=int,
        help="Number of epochs that should be used for the FNN training",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-m",
        "--enableMultiLabel",
        metavar="BOOL",
        type=bool,
        help="Train multi-label classification model in addition to the individual models.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-l",
        "--testSize",
        metavar="INT",
        type=float,
        help="Fraction of the dataset that should be used for testing. Value in [0,1].",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-K",
        "--kFolds",
        metavar="INT",
        type=int,
        help="K that is used for K-fold cross validation in the training procedure.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        metavar="INT",
        type=int,
        choices=[0, 1, 2],
        help="Verbosity level. O: No additional output, "
        + "1: Some additional output, 2: full additional output",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trainAC",
        metavar="BOOL",
        type=bool,
        help="Train the autoencoder based on the input file and store the respective weights to the "
        "file provided with -a option. Set this to False, if the file provided"
        "with -a option already contains the weights of the autoencoder you would like to use.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trainFNN",
        metavar="BOOL",
        type=bool,
        help="Train the feed forward network either with provided weights of a trained autoencoder"
        "(see option -a and --trainAC=False), or train the autoencoder prior to the training"
        "of the feed forward network(s).",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sampleFractionOnes",
        metavar="FLOAT",
        type=float,
        help='This is the fraction of positive target associations (1s) after sampling from the "'
        "negative target association (0s).",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sampleDown",
        metavar="BOOL",
        type=bool,
        help="Enable automatic down sampling of the 0 valued samples to compensate extremely "
        "unbalanced data (<10%%).",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--lossFunction",
        metavar="character",
        type=str,
        choices=["mse", "bce"],
        help="Loss function to use during training. "
        "mse - mean squared error, bce - binary cross entropy.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--optimizer",
        metavar="character",
        type=str,
        choices=["Adam", "SGD"],
        help='Optimizer to use for backpropagation in the FNN. Possible values: "Adam", "SGD"',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batchSize",
        metavar="INT",
        type=int,
        help="Batch size in FNN training.",
        default=argparse.SUPPRESS,
    )

    # Specific options for AE training
    parser.add_argument(
        "--aeEpochs",
        metavar="INT",
        type=int,
        help="Number of epochs for autoencoder training.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aeBatchSize",
        metavar="INT",
        type=int,
        help="Batch size in FNN training.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aeActivationFunction",
        metavar="STRING",
        type=str,
        choices=["relu", "tanh", "selu", "elu"],
        help="The activation function for hidden layers in the autoencoder.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aeLearningRate",
        metavar="FLOAT",
        type=float,
        help="Learning rate for AC training.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aeLearningRateDecay",
        metavar="FLOAT",
        type=float,
        help="Learning rate decay for AC training.",
        default=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--learningRate",
        metavar="FLOAT",
        type=float,
        help="Batch size in FNN training.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--activationFunction",
        metavar="character",
        type=str,
        choices=["relu", "tanh", "selu", "elu", "exponential"],
        help="The activation function for hidden layers in the FNN.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--l2reg",
        metavar="FLOAT",
        type=float,
        help="Value for l2 kernel regularizer.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dropout",
        metavar="FLOAT",
        type=float,
        help="The fraction of data that is dropped out in each dropout layer.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wabTracking",
        metavar="STRING",
        type=str,
        help="Track FNN performance via Weights & Biases, see https://wandb.ai.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aeWabTracking",
        metavar="STRING",
        type=str,
        help="Track autoencoder performance via Weights & Biases, see https://wandb.ai.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wabTarget",
        metavar="STRING",
        type=str,
        choices=["AR", "ER", "ED", "GR", "TR", "PPARg", "Aromatase"],
        help="Which target to use for tracking performance via Weights & Biases, see https://wandb.ai.",
        default=argparse.SUPPRESS,
    )


def parseTrainGnn(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--save", type=bool)
    parser.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="Input JSON file that contains all information for training/predicting.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        choices=list(range(torch.cuda.device_count())),
        help="Which GPU to use",
    )
    parser.add_argument(
        "--data_path", type=str, help="Path to data CSV file", default=""
    )
    parser.add_argument(
        "--use_compound_names",
        action="store_true",
        default=False,
        help="Use when test data file contains compound names in addition to SMILES strings",
    )
    parser.add_argument(
        "--max_data_size", type=int, help="Maximum number of data points to load"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether to skip training and only test the model",
    )
    parser.add_argument(
        "--features_only",
        action="store_true",
        default=False,
        help="Use only the additional features in an FFN, no graph network",
    )
    # parser.add_argument('--features_generator', type=str, nargs='*',
    #                     choices=get_available_features_generators(),
    #                     help='Method of generating additional features')
    parser.add_argument(
        "--features_path",
        type=str,
        nargs="*",
        help="Path to features to use in FNN (instead of features_generator)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./ckpt/",
        help="Directory where model checkpoints will be saved",
    )
    parser.add_argument(
        "--save_smiles_splits",
        action="store_true",
        default=False,
        help="Save smiles for each train/val/test splits for prediction convenience later",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory from which to load model checkpoints"
        "(walks directory and ensembles all models that are found)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="*",
        default=None,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["classification", "regression", "multiclass"],
        help="Type of dataset, e.g. classification or regression."
        "This determines the loss function used during training.",
        default="regression",
    )  # classification
    parser.add_argument(
        "--multiclass_num_classes",
        type=int,
        default=3,
        help="Number of classes when running multiclass classification",
    )
    parser.add_argument(
        "--separate_val_path", type=str, help="Path to separate val set, optional"
    )
    parser.add_argument(
        "--separate_val_features_path",
        type=str,
        nargs="*",
        help="Path to file with features for separate val set",
    )
    parser.add_argument(
        "--separate_test_path", type=str, help="Path to separate test set, optional"
    )
    parser.add_argument(
        "--separate_test_features_path",
        type=str,
        nargs="*",
        help="Path to file with features for separate test set",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--split_sizes",
        type=float,
        nargs=3,
        default=[0.8, 0.2, 0.0],
        help="Split proportions for train/validation/test sets",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=1,
        help="Number of folds when performing cross validation",
    )
    parser.add_argument(
        "--folds_file", type=str, default=None, help="Optional file of fold labels"
    )
    parser.add_argument(
        "--val_fold_index",
        type=int,
        default=None,
        help="Which fold to use as val for leave-one-out cross val",
    )
    parser.add_argument(
        "--test_fold_index",
        type=int,
        default=None,
        help="Which fold to use as test for leave-one-out cross val",
    )
    parser.add_argument(
        "--crossval_index_dir",
        type=str,
        help="Directory in which to find cross validation index files",
    )
    parser.add_argument(
        "--crossval_index_file",
        type=str,
        help="Indices of files to use as train/val/test"
        "Overrides --num_folds and --seed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed to use when splitting data into train/val/test sets."
        "When `num_folds` > 1, the first fold uses this seed and all"
        "subsequent folds add 1 to the seed.",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Skip non-essential print statements",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=10,
        help="The number of batches between each logging of the training loss",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=True, help="Turn off cuda"
    )
    parser.add_argument(
        "--show_individual_scores",
        action="store_true",
        default=True,
        help="Show all scores for individual targets, not just average, at the end",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        default=False,
        help="Turn off caching mol2graph computation",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to a .json file containing arguments. Any arguments present in the config"
        "file will override arguments specified via the command line or by the defaults.",
    )
    parser.add_argument("--smiles_columns", type=str, help="Name of the smiles columns")

    parser.add_argument("--target_columns", type=str, help="Name of the target columns")

    parser.add_argument(
        "--ignore_columns", type=str, help="Ignore or not the smiles columns"
    )
    parser.add_argument("--num_tasks", type=int, help="NUmber of tasks")
    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to run"
    )
    parser.add_argument(
        "--total_epochs", type=int, default=30, help="Number of total epochs to run"
    )
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=2.0,
        help="Number of epochs during which learning rate increases linearly from"
        "init_lr to max_lr. Afterwards, learning rate decreases exponentially"
        "from max_lr to final_lr.",
    )
    parser.add_argument(
        "--init_lr", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--max_lr", type=float, default=1e-3, help="Maximum learning rate"
    )
    parser.add_argument(
        "--final_lr", type=float, default=1e-4, help="Final learning rate"
    )
    parser.add_argument(
        "--no_features_scaling",
        action="store_true",
        default=False,
        help="Turn off scaling of features",
    )
    parser.add_argument(
        "--features_scaling",
        action="store_true",
        default=False,
        help="Turn on scaling of features",
    )
    parser.add_argument(
        "--use_input_features", type=str, help="Turn on scaling of features"
    )
    parser.add_argument("--extra_metrics", nargs="*", help="Extra metrics to use")
    # Model arguments
    parser.add_argument(
        "--ensemble_size", type=int, default=1, help="Number of models in ensemble"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=300,
        help="Dimensionality of hidden layers in MPN",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Whether to add bias to linear layers",
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Number of message passing steps"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="ReLU",
        choices=["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"],
        help="Activation function",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        default=False,
        help="Undirected edges (always sum the two relevant bond vectors)",
    )
    parser.add_argument(
        "--ffn_hidden_size",
        type=int,
        default=2,
        help="Hidden dim for higher-capacity FFN (defaults to hidden_size)",
    )
    parser.add_argument(
        "--ffn_num_layers",
        type=int,
        default=2,
        help="Number of layers in FFN after MPN encoding",
    )
    parser.add_argument(
        "--atom_messages",
        action="store_true",
        default=False,
        help="Use messages on atoms instead of messages on bonds",
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="Turn on cuda"
    )
    parser.add_argument(
        "--num_lrs",
        type=int,
        default=2,
        help="Number of layers in FFN after MPN encoding",
    )
    parser.add_argument(
        "--minimize_score",
        type=bool,
        default=False,
        help="Number of layers in FFN after MPN encoding",
    )
    parser.add_argument(
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
    parser.add_argument("--data_weights_path", type=str)
    # parser.add_argument("--target_weights": List[float])
    parser.add_argument("--split_key_molecule", type=int)
    parser.add_argument("--pytorch_seed", type=int)
    parser.add_argument("--checkpoint_frzn", type=str)
    parser.add_argument("--cache_cutoff", type=float)
    parser.add_argument("--save_preds", type=bool)
    # Model arguments
    parser.add_argument("--mpn_shared", type=bool)
    parser.add_argument("--separate_val_phase_features_path", type=str)
    parser.add_argument("--separate_test_phase_features_path", type=str)

    parser.add_argument("--separate_val_atom_descriptors_path", type=str)
    parser.add_argument("--separate_test_atom_descriptors_path", type=str)
    parser.add_argument("--aggregation", choices=["mean", "sum", "norm"])
    parser.add_argument("--aggregation_norm", type=int)
    parser.add_argument("--explicit_h", type=bool)
    parser.add_argument("--adding_h", type=bool)
    # Training arguments
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--class_balance", type=bool)
    parser.add_argument("--evidential_regularization", type=float)
    parser.add_argument("--overwrite_default_atom_features", type=bool)
    parser.add_argument("--no_atom_descriptor_scaling", type=bool)
    parser.add_argument("--overwrite_default_bond_features", type=bool)
    parser.add_argument("--frzn_ffn_layers", type=int)
    parser.add_argument("--freeze_first_only", type=bool)
    parser.add_argument(
        "-gt",
        "--gnn_type",
        metavar="STR",
        choices=["cmpnn", "dmpnn"],
        help="Define GNN Model",
        default=argparse.SUPPRESS,
    )


def parseInputPredict(parser: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    #    parser = argparse.ArgumentParser(description='Use models that have been generated by deepFPlearn-Train.py '
    #                                                 'tool to make predictions on chemicals (provide SMILES or '
    #                                                 'topological fingerprints).')

    parser.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="Input JSON file that contains all information for training/predicting.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
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
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        metavar="FILE",
        type=str,
        help="Prefix of output directory. It will contain a log file and the file specified"
        "with --outputFile.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--outputFile",
        metavar="FILE",
        type=str,
        help="Output .CSV file name which will contain one prediction per input line. "
        "Default: prefix of input file name.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-t",
        "--type",
        metavar="STR",
        type=str,
        choices=["fp", "smiles"],
        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-k",
        "--fpType",
        metavar="STR",
        type=str,
        choices=["topological", "MACCS"],  # , 'atompairs', 'torsions'],
        help="The type of fingerprint to be generated/used in input file.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ecModelDir",
        type=str,
        metavar="DIR",
        help="The directory where the full model of the encoder is loaded from."
        "Provide a full path here.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fnnModelDir",
        type=str,
        metavar="DIR",
        help="The directory where the full model of the fnn is loaded from. "
        "Provide a full path here.",
        default=argparse.SUPPRESS,
    )


def parsePredictGnn(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--configFile",
        metavar="FILE",
        type=str,
        help="Input JSON file that contains all information for training/predicting.",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        choices=list(range(torch.cuda.device_count())),
        help="Which GPU to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to CSV file containing testing data for which predictions will be made",
        default="",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        help="Path to CSV file containing testing data for which predictions will be made",
        default="",
    )
    parser.add_argument(
        "--use_compound_names",
        action="store_true",
        default=False,
        help="Use when test data file contains compound names in addition to SMILES strings",
    )
    parser.add_argument(
        "--preds_path",
        type=str,
        help="Path to CSV file where predictions will be saved",
        default="",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory from which to load model checkpoints"
        "(walks directory and ensembles all models that are found)",
        default="./ckpt",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="*",
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="Turn off cuda"
    )
    # parser.add_argument('--features_generator', type=str, nargs='*',
    #                     choices=get_available_features_generators(),
    #                     help='Method of generating additional features')
    parser.add_argument(
        "--features_path",
        type=str,
        nargs="*",
        help="Path to features to use in FNN (instead of features_generator)",
    )
    parser.add_argument(
        "--no_features_scaling",
        action="store_true",
        default=False,
        help="Turn off scaling of features",
    )
    parser.add_argument(
        "--max_data_size", type=int, help="Maximum number of data points to load"
    )
    parser.add_argument(
        "--smiles_columns",
        type=str,
        help="List of names of the columns containing SMILES strings.By default, uses the first\
                             number_of_molecules columns.",
    )
    parser.add_argument(
        "--number_of_molecules",
        type=int,
        help="Number of molecules in each input to the model.This must equal the length of\
                             smiles_columns if not None",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers for the parallel data loading 0 means sequential",
    )
    parser.add_argument(
        "--atom_descriptors", type=str, help="Custom extra atom descriptors"
    )
    parser.add_argument(
        "--atom_descriptors_path", type=str, help="Path to the extra atom descriptors."
    )
    parser.add_argument(
        "--bond_features_size",
        type=str,
        help="Size of the extra bond descriptors that will be used as bond features to featurize a\
                             given molecule",
    )
    parser.add_argument(
        "--no_cache",
        type=bool,
        default=False,
        help="Turn off caching mol2graph computation",
    )
    parser.add_argument(
        "--no_cache_mol",
        type=bool,
        default=False,
        help="Whether to not cache the RDKit molecule for each SMILES string to reduce memory\
                             usage cached by default",
    )
    parser.add_argument(
        "--empty_cache",
        type=bool,
        help="Whether to empty all caches before training or predicting. This is necessary if\
                             multiple jobs are run within a single script and the atom or bond features change",
    )
    parser.add_argument(
        "-gt",
        "--gnn_type",
        metavar="STR",
        choices=["cmpnn", "dmpnn"],
        help="Define GNN Model",
        default=argparse.SUPPRESS,
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
