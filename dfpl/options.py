from __future__ import annotations
from dataclasses import dataclass
import jsonpickle
import argparse
from pathlib import Path

from dfpl.utils import makePathAbsolute


@dataclass
class Options:
    """
    Dataclass for all options necessary for training the neural nets
    """
    configFile: str = ""
    inputFile: str = "../data/input_datasets/S_dataset.csv"
    outputDir: str = "data/output/S_dataset"
    outputFile: str = ""
    ecWeightsFile: str = "AE.encoder.weights.hdf5"
    ecModelDir: str = 'AE_encoder'
    fnnModelDir: str = "modeltraining"
    type: str = "smiles"
    fpType: str = "topological"  # also "MACCS", "atompairs"

    epochs: int = 50
    fpSize: int = 2048
    encFPSize: int = 256
    kFolds: int = 1
    testSize: float = 0.2
    enableMultiLabel: bool = False
    verbose: int = 0
    trainAC: bool = True  # if set to False, an AC weight file must be provided!
    trainFNN: bool = True
    compressFeatures: bool = True
    sampleFractionOnes: float = 0.5  # Only used when value is in [0,1]
    sampleDown: bool = False

    aeEpochs: int = 30
    aeBatchSize: int = 512
    aeLearningRate: float = 0.001
    aeLearningRateDecay: float = 0.01
    aeActivationFunction: str = 'relu'

    fnnType: str = "FNN"
    batchSize: int = 128
    optimizer: str = "Adam"
    learningRate: float = 0.001
    lossFunction: str = "bce"
    activationFunction: str = "relu"
    l2reg: float = 0.001
    dropout: float = 0.2
    snnDepth = 8
    snnWidth = 50
    wabTracking: bool = True  # Wand & Biases tracking
    wabTarget: str = "ARR"  # Wand & Biases target used for showing training progress

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
                raise ValueError(f"Could not find JSON input file: {jsonFile}")

        for key, value in vars(args).items():
            # The args dict will contain a "method" key from the subparser.
            # We don't use this.
            if key != "method":
                result.__setattr__(key, value)
        return result


def createCommandlineParser() -> argparse.ArgumentParser:
    """
    Build the parser for arguments with its two subparsers
    """
    parser = argparse.ArgumentParser(prog='deepFPlearn')
    subparsers = parser.add_subparsers(help="Sub programs of deepFPlearn")

    parser_train = subparsers.add_parser("train", help="Train new models with your data")
    parser_train.set_defaults(method="train")
    parseInputTrain(parser_train)

    parser_predict = subparsers.add_parser("predict", help="Predict your data with existing models")
    parser_predict.set_defaults(method="predict")
    parseInputPredict(parser_predict)

    parser_convert = subparsers.add_parser("convert", help="Convert known data files to pickle serialization files")
    parser_convert.set_defaults(method="convert")
    parseInputConvert(parser_convert)
    return parser


def parseInputTrain(parser: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    #    parser = argparse.ArgumentParser(description='Train a DNN to associate chemical fingerprints with a (set of)'
    #                                     'target(s). Trained models are saved to disk including fitted weights and '
    #                                     'can be used in the deepFPlearn-Predict.py tool to make predictions.')
    parser.add_argument("-f", "--configFile",
                        metavar='FILE',
                        type=str,
                        help="Input JSON file that contains all information for training/predicting.",
                        default=argparse.SUPPRESS)
    parser.add_argument("-i", "--inputFile",
                        metavar='FILE',
                        type=str,
                        help="The file containing the data for training in (unquoted) "
                             "comma separated CSV format. First column contain the feature string in "
                             "form of a fingerprint or a SMILES (see -t option). "
                             "The remaining columns contain the outcome(s) (Y matrix). "
                             "A header is expected and respective column names are used "
                             "to refer to outcome(s) (target(s)).",
                        default=argparse.SUPPRESS)
    parser.add_argument("-o", "--outputDir",
                        metavar='FILE',
                        type=str,
                        help='Prefix of output file name. Trained model(s) and '
                             'respective stats will be returned in this directory.',
                        default=argparse.SUPPRESS)
    parser.add_argument('-t', "--type",
                        metavar='STR',
                        type=str,
                        choices=['fp', 'smiles'],
                        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
                        default=argparse.SUPPRESS)
    parser.add_argument('-k', "--fpType",
                        metavar='STR',
                        type=str,
                        choices=['topological', 'MACCS'],  # , 'atompairs', 'torsions'],
                        help='The type of fingerprint to be generated/used in input file.',
                        default=argparse.SUPPRESS)
    parser.add_argument('-s', "--fpSize",
                        type=int,
                        help='Size of fingerprint that should be generated.',
                        default=argparse.SUPPRESS)
    parser.add_argument('-c', "--compressFeatures",
                        metavar='BOOL',
                        type=bool,
                        help='Compress the fingerprints. This is done either with an existing autoencoder or a new '
                             'autoencoder model will be trained using the input compounds (see further options for '
                             'details).',
                        default=argparse.SUPPRESS)
    parser.add_argument("--fnnType",
                        metavar='STR',
                        type=str,
                        choices=['FNN', 'SNN', 'REG'],
                        help='Define the DL model architecture 2B used.',
                        default=argparse.SUPPRESS)
    parser.add_argument('-a', "--ecWeightsFile",
                        type=str,
                        metavar='FILE',
                        help='The .hdf5 file of a trained encoder (e.g. from a previous'
                             'training run. This avoids a retraining of the autoencoder on the'
                             'training data set (provided with -i). NOTE that the input and encoding'
                             'dimensions must fit your data and settings. Default: train new autoencoder.',
                        default=argparse.SUPPRESS)
    parser.add_argument("--ecModelDir",
                        type=str,
                        metavar='DIR',
                        help='The directory where the full model of the encoder will be saved (if trainAE=True) or '
                             'loaded from (if trainAE=False). Provide a full path here.',
                        default=argparse.SUPPRESS)
    parser.add_argument("-d", "--encFPSize",
                        metavar='INT',
                        type=int,
                        help='Size of encoded fingerprint (z-layer of autoencoder).',
                        default=argparse.SUPPRESS)
    parser.add_argument("-e", "--epochs",
                        metavar='INT', type=int,
                        help='Number of epochs that should be used for the FNN training',
                        default=argparse.SUPPRESS)
    parser.add_argument('-m', "--enableMultiLabel",
                        metavar="BOOL",
                        type=bool,
                        help='Train multi-label classification model in addition to the individual models.',
                        default=argparse.SUPPRESS)
    parser.add_argument('-l', "--testSize",
                        metavar='INT',
                        type=float,
                        help='Fraction of the dataset that should be used for testing. Value in [0,1].',
                        default=argparse.SUPPRESS)
    parser.add_argument("-K", "--kFolds",
                        metavar='INT',
                        type=int,
                        help='K that is used for K-fold cross validation in the training procedure.',
                        default=argparse.SUPPRESS)
    parser.add_argument("-v", "--verbose",
                        metavar='INT',
                        type=int,
                        choices=[0, 1, 2],
                        help="Verbosity level. O: No additional output, " +
                             "1: Some additional output, 2: full additional output",
                        default=argparse.SUPPRESS)
    parser.add_argument('--trainAC',
                        metavar='BOOL',
                        type=bool,
                        help='Train the autoencoder based on the input file and store the respective weights to the '
                             'file provided with -a option. Set this to False, if the file provided'
                             'with -a option already contains the weights of the autoencoder you would like to use.',
                        default=argparse.SUPPRESS)
    parser.add_argument('--trainFNN',
                        metavar='BOOL',
                        type=bool,
                        help='Train the feed forward network either with provided weights of a trained autoencoder'
                             '(see option -a and --trainAC=False), or train the autoencoder prior to the training'
                             'of the feed forward network(s).',
                        default=argparse.SUPPRESS)
    parser.add_argument('--sampleFractionOnes',
                        metavar='FLOAT',
                        type=float,
                        help='This is the fraction of positive target associations (1s) after sampling from the "'
                             'negative target association (0s).',
                        default=argparse.SUPPRESS)
    parser.add_argument('--sampleDown',
                        metavar='BOOL',
                        type=bool,
                        help='Enable automatic down sampling of the 0 valued samples to compensate extremely '
                             'unbalanced data (<10%%).',
                        default=argparse.SUPPRESS)
    parser.add_argument('--lossFunction',
                        metavar="character",
                        type=str,
                        choices=["rmse", "mse", "bce"],
                        help="Loss function to use during training. "
                             "mse - mean squared error, bce - binary cross entropy.",
                        default=argparse.SUPPRESS)
    parser.add_argument('--optimizer',
                        metavar="character",
                        type=str,
                        choices=["Adam", "SGD"],
                        help="Optimizer to use for backpropagation in the FNN. Possible values: \"Adam\", \"SGD\"",
                        default=argparse.SUPPRESS)
    parser.add_argument('--batchSize',
                        metavar="INT",
                        type=int,
                        help="Batch size in FNN training.",
                        default=argparse.SUPPRESS)

    # Specific options for AE training
    parser.add_argument('--aeEpochs',
                        metavar="INT",
                        type=int,
                        help="Number of epochs for autoencoder training.",
                        default=argparse.SUPPRESS)
    parser.add_argument('--aeBatchSize',
                        metavar="INT",
                        type=int,
                        help="Batch size in FNN training.",
                        default=argparse.SUPPRESS)
    parser.add_argument('--aeActivationFunction',
                        metavar="STRING",
                        type=str,
                        choices=["relu", "tanh", "selu", "elu", "smht"],
                        help="The activation function for hidden layers in the autoencoder.",
                        default=argparse.SUPPRESS)
    parser.add_argument('--aeLearningRate',
                        metavar="FLOAT",
                        type=float,
                        help="Learning rate for AC training.",
                        default=argparse.SUPPRESS)
    parser.add_argument('--aeLearningRateDecay',
                        metavar="FLOAT",
                        type=float,
                        help="Learning rate decay for AC training.",
                        default=argparse.SUPPRESS)

    parser.add_argument('--learningRate',
                        metavar="FLOAT",
                        type=float,
                        help="Batch size in FNN training.",
                        default=argparse.SUPPRESS)
    parser.add_argument('--activationFunction',
                        metavar="character",
                        type=str,
                        choices=["relu", "tanh", "selu", "elu", "exponential"],
                        help="The activation function for hidden layers in the FNN.",
                        default=argparse.SUPPRESS)
    parser.add_argument('--l2reg',
                        metavar="FLOAT",
                        type=float,
                        help="Value for l2 kernel regularizer.",
                        default=argparse.SUPPRESS)
    parser.add_argument('--dropout',
                        metavar="FLOAT",
                        type=float,
                        help="The fraction of data that is dropped out in each dropout layer.",
                        default=argparse.SUPPRESS)
    parser.add_argument('--wabTracking',
                        metavar="BOOL",
                        type=bool,
                        help="Track training performance via Weights & Biases, see https://wandb.ai.",
                        default=argparse.SUPPRESS)
    parser.add_argument('--wabTarget',
                        metavar="STRING",
                        type=str,
                        help="Which target to use for tracking training performance via Weights & Biases, "
                             "see https://wandb.ai.",
                        default=argparse.SUPPRESS)


# @dataclass
# class PredictOptions(Options):
#     """
#     Dataclass to hold all options used for prediction
#     """
#
#     def saveToFile(self, file: str) -> None:
#         """
#         Export an instance to JSON. This file is useful for creating template JSON files
#         """
#         jsonFile = Path(file)
#         with jsonFile.open("w") as f:
#             f.write(jsonpickle.encode(self))
#
#     @classmethod
#     def fromJson(cls, file: str) -> PredictOptions:
#         """
#         Create an instance from a JSON file
#         """
#         jsonFile = Path(file)
#         if jsonFile.exists() and jsonFile.is_file():
#             with jsonFile.open() as f:
#                 content = f.read()
#                 return jsonpickle.decode(content)
#         raise ValueError("JSON file does not exist or is not readable")
#
#     @classmethod
#     def fromCmdArgs(cls, args: argparse.Namespace) -> PredictOptions:
#         """Creates Options instance from cmdline arguments"""
#         for key, value in vars(args).items():
#             # The args dict will contain a "method" key from the subparser.
#             # We don't use this.
#             if key != "method":
#                 result.__setattr__(key, value)
#         return result
#
#         if args.f != "":
#             jsonFile = Path(makePathAbsolute(args.f))
#             if jsonFile.exists() and jsonFile.is_file():
#                 with jsonFile.open() as f:
#                     content = f.read()
#                     return jsonpickle.decode(content)
#             else:
#                 raise ValueError("Could not find JSON input file")
#         else:
#             return cls(
#                 inputFile=args.i,
#                 outputDir=args.o,
#                 ecWeightsFile=args.ECmodel,
#                 model=args.model,
#                 target=args.target,
#                 fpSize=args.s,
#                 encFPSize=args.d,
#                 type=args.t,
#                 fpType=args.k,
#             )


def parseInputPredict(parser: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    #    parser = argparse.ArgumentParser(description='Use models that have been generated by deepFPlearn-Train.py '
    #                                                 'tool to make predictions on chemicals (provide SMILES or '
    #                                                 'topological fingerprints).')

    parser.add_argument("-f", "--configFile",
                        metavar='FILE',
                        type=str,
                        help="Input JSON file that contains all information for training/predicting.",
                        default=argparse.SUPPRESS)
    parser.add_argument("-i", "--inputFile",
                        metavar='FILE',
                        type=str,
                        help="The file containing the data for the prediction in (unquoted) "
                             "comma separated CSV format. The column named 'smiles' or 'fp'"
                             "contains the field to be predicted. Please adjust the type "
                             "that should be predicted (fp or smile) with -t option appropriately."
                             "An optional column 'id' is used to assign the outcomes to the"
                             "original identifiers. If this column is missing, the results are"
                             "numbered in the order of their appearance in the input file."
                             "A header is expected and respective column names are used.",
                        default=argparse.SUPPRESS)
    parser.add_argument("-o", "--outputDir",
                        metavar='FILE',
                        type=str,
                        help='Prefix of output directory. It will contain a log file and the file specified'
                             'with --outputFile.',
                        default=argparse.SUPPRESS)
    parser.add_argument("--outputFile",
                        metavar='FILE',
                        type=str,
                        help='Output .CSV file name which will contain one prediction per input line. '
                             'Default: prefix of input file name.',
                        default=argparse.SUPPRESS)
    parser.add_argument('-t', "--type",
                        metavar='STR',
                        type=str,
                        choices=['fp', 'smiles'],
                        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
                        default=argparse.SUPPRESS)
    parser.add_argument('-k', "--fpType",
                        metavar='STR',
                        type=str,
                        choices=['topological', 'MACCS'],  # , 'atompairs', 'torsions'],
                        help='The type of fingerprint to be generated/used in input file.',
                        default=argparse.SUPPRESS)
    parser.add_argument("--ecModelDir",
                        type=str,
                        metavar='DIR',
                        help='The directory where the full model of the encoder is loaded from.'
                             'Provide a full path here.',
                        default=argparse.SUPPRESS)
    parser.add_argument("--fnnModelDir",
                        type=str,
                        metavar='DIR',
                        help='The directory where the full model of the fnn is loaded from. '
                             'Provide a full path here.',
                        default=argparse.SUPPRESS)


def parseInputConvert(parser: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    parser.add_argument('-f', metavar='FILE', type=str,
                        help="Input directory where your CSV/TSV files are stored.",
                        required=True, default="")
