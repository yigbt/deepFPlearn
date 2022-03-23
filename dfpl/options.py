from __future__ import annotations
from dataclasses import dataclass
import jsonpickle
import argparse
from pathlib import Path

from dfpl.utils import makePathAbsolute


@dataclass
class TrainOptions:
    """
    Dataclass for all options necessary for training the neural nets
    """
    inputFile: str = "data/Sun_etal_dataset.csv"
    outputDir: str = "modeltraining"
    ecWeightsFile: str = "Sun_etal_dataset.AC.encoder.weights.hdf5"
    type: str = "smiles"
    fpType: str = "topological"  # also "MACCS", "atompairs"
    epochs: int = 512
    fpSize: int = 2048
    encFPSize: int = 256
    kFolds: int = 0
    testSize: float = 0.2
    enableMultiLabel: bool = True
    verbose: int = 0
    trainAC: bool = True  # if set to False, an AC weight file must be provided!
    trainFNN: bool = True
    compressFeatures: bool = True
    sampleFractionOnes: float = None
    sampleDown: bool = False
    lossFunction: str = "mse"
    optimizer: str = "Adam"
    batchSize: int = 128
    learningRate: float = 0.001
    activationFunction: str = "relu"
    l2reg: float = 0.001
    dropout: float = 0.2
    wabTracking: bool = False  # Wand & Biases tracking
    wabTarget: str = "ER"  # Wand & Biases target used for showing training progress

    def saveToFile(self, file: str) -> None:
        """
        Saves an instance to a JSON file
        """
        jsonFile = Path(file)
        with jsonFile.open("w") as f:
            f.write(jsonpickle.encode(self))

    @classmethod
    def fromJson(cls, file: str) -> TrainOptions:
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
    def fromCmdArgs(cls, args: argparse.Namespace) -> TrainOptions:
        """Creates TrainOptions instance from cmdline arguments"""
        if args.f != "":
            jsonFile = Path(makePathAbsolute(args.f))
            if jsonFile.exists() and jsonFile.is_file():
                with jsonFile.open() as f:
                    content = f.read()
                    return jsonpickle.decode(content)
            else:
                raise ValueError("Could not find JSON input file")
        else:
            return cls(
                inputFile=args.i,
                outputDir=args.o,
                ecWeightsFile=args.a,
                type=args.t,
                fpType=args.k,
                fpSize=args.s,
                encFPSize=args.d,
                epochs=args.e,
                kFolds=args.K,
                testSize=args.l,
                enableMultiLabel=args.m,
                verbose=args.v,
                trainAC=args.trainAC,
                trainFNN=args.trainFNN,
                compressFeatures=args.c,
                sampleFractionOnes=args.sampleFractionOnes,
                sampleDown=args.sampleDown,
                lossFunction=args.lossFunction,
                optimizer=args.optimizer,
                batchSize=args.batchSize,
                learningRate=args.learningRate,
                activationFunction=args.activationFunction,
                l2reg=args.l2reg,
                dropout=args.dropout,
                wabTracking=args.wabTracking,
                wabTarget=args.wabTarget
            )


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
    parser.add_argument('-f', metavar='FILE', type=str,
                        help="Input JSON file that contains all information for training.",
                        required=False, default="")
    parser.add_argument('-i', metavar='FILE', type=str,
                        help="The file containing the data for training in (unquoted) "
                             "comma separated CSV format. First column contain the feature string in "
                             "form of a fingerprint or a SMILES (see -t option). "
                             "The remaining columns contain the outcome(s) (Y matrix). "
                             "A header is expected and respective column names are used "
                             "to refer to outcome(s) (target(s)).", required=False)
    parser.add_argument('-o', metavar='FILE', type=str,
                        help='Prefix of output file name. Trained model(s) and '
                             'respective stats will be returned in 2 output files '
                             'with this prefix. Default: prefix of input file name.')
    parser.add_argument('-t', metavar='STR', type=str, choices=['fp', 'smiles'],
                        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
                        default='smiles')
    parser.add_argument('-k', metavar='STR', type=str,
                        choices=['topological', 'MACCS'],  # , 'atompairs', 'torsions'],
                        help='The type of fingerprint to be generated/used in input file.',
                        default='topological')
    parser.add_argument('-s', type=int,
                        help='Size of fingerprint that should be generated.',
                        default=2048)
    parser.add_argument('-c', metavar='BOOL', type=bool,
                        help='Compress the fingerprints. This is done either with an existing autoencoder or a new '
                             'autoencoder model will be trained using the input compounds (see further options for '
                             'details).',
                        default=True)
    parser.add_argument('-a', type=str, metavar='FILE', default=None,
                        help='The .hdf5 file of a trained encoder (e.g. from a previous'
                             'training run. This avoids a retraining of the autoencoder on the'
                             'training data set (provided with -i). NOTE that the input and encoding'
                             'dimensions must fit your data and settings. Default: train new autoencoder.')
    parser.add_argument('--ACtrainOnly', action='store_true',
                        help='Only train the autoencoder on the features matrix an exit.')
    parser.add_argument('-d', metavar='INT', type=int,
                        help='Size of encoded fingerprint (z-layer of autoencoder).',
                        default=256)
    parser.add_argument('-e', metavar='INT', type=int,
                        help='Number of epochs that should be trained',
                        default=50)
    parser.add_argument('-p', metavar='FILE', type=str,
                        help="CSV file containing the parameters for the epochs per target model."
                             "The target abbreviation should be the same as in the input file and"
                             "the columns/parameters are: \n\ttarget,batch_size,epochs,optimizer,activation."
                             "Note that values in from file overwrite -e option!")
    parser.add_argument('-m', action='store_true',
                        help='Train multi-label classification model in addition to the individual models.')
    parser.add_argument('-l', metavar='INT', type=float,
                        help='Fraction of the dataset that should be used for testing. Value in [0,1].',
                        default=0.2)
    parser.add_argument('-K', metavar='INT', type=int,
                        help='K that is used for K-fold cross validation in the training procedure.',
                        default=5)
    parser.add_argument('-v', metavar='INT', type=int, choices=[0, 1, 2],
                        help="Verbosity level. O: No additional output, " +
                             "1: Some additional output, 2: full additional output",
                        default=2
                        )
    parser.add_argument('--trainAC', metavar='BOOL', type=bool,
                        help='Train the autoencoder based on the input file and store the respective weights to the '
                             'file provided with -a option. Set this to False, if the file provided'
                             'with -a option already contains the weights of the autoencoder you would like to use.',
                        default=True)
    parser.add_argument('--trainFNN', metavar='BOOL', type=bool,
                        help='Train the feed forward network either with provided weights of a trained autoencoder'
                             '(see option -a and --trainAC=False), or train the autoencoder prior to the training'
                             'of the feed forward network(s).',
                        default=True)
    parser.add_argument('--sampleFractionOnes', metavar='FLOAT', type=float,
                        help='This is the fraction of positive target associations (1s) after sampling from the "'
                             'negative target association (0s).',
                        default=None)
    parser.add_argument('--sampleDown', metavar='BOOL', type=bool,
                        help='Enable automatic down sampling of the 0 valued samples to compensate extremely '
                             'unbalanced data (<10%%).',
                        default=False)
    parser.add_argument('--lossFunction', metavar="character", type=str,
                        choices=["mse", "bce"], default="mse",
                        help="Loss function to use during training. "
                             "mse - mean squared error, bce - binary cross entropy.")
    parser.add_argument('--optimizer', metavar="character", type=str,
                        choices=["Adam", "SGD"], default="mse",
                        help="Optimizer to use for backpropagation in the FNN.")
    parser.add_argument('--batchSize', metavar="INT", type=int,
                        default=128,
                        help="Batch size in FNN training.")
    parser.add_argument('--learningRate', metavar="FLOAT", type=float,
                        default=0.001,
                        help="Batch size in FNN training.")
    parser.add_argument('--activationFunction', metavar="character", type=str,
                        default="relu", choices=["relu", "tanh", "selu", "elu", "exponential"],
                        help="The activation function for hidden layers in the FNN.")
    parser.add_argument('--l2reg', metavar="FLOAT", type=float,
                        default=0.001,
                        help="Value for l2 kernel regularizer.")
    parser.add_argument('--dropout', metavar="FLOAT", type=float,
                        default=0.2,
                        help="The fraction of data that is dropped out in each dropout layer.")
    parser.add_argument('--wabTracking', metavar="BOOL", type=bool,
                        default=False,
                        help="Track training performance via Weights & Biases, see https://wandb.ai.")
    parser.add_argument('--wabTarget', metavar="STRING", type=str,
                        default="ER",
                        help="Which target to use for tracking training performance via Weights & Biases, "
                             "see https://wandb.ai.")


@dataclass
class PredictOptions:
    """
    Dataclass to hold all options used for prediction
    """
    inputFile: str = ""
    outputDir: str = ""
    ecWeightsFile: str = ""
    model: str = ""
    target: str = ""
    fpSize: int = 2048
    encFPSize: int = 256
    type: str = "smiles"
    fpType: str = "topological"  # also "MACCS", "atompairs"

    def saveToFile(self, file: str) -> None:
        """
        Export an instance to JSON. This file is useful for creating template JSON files
        """
        jsonFile = Path(file)
        with jsonFile.open("w") as f:
            f.write(jsonpickle.encode(self))

    @classmethod
    def fromJson(cls, file: str) -> PredictOptions:
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
    def fromCmdArgs(cls, args: argparse.Namespace) -> PredictOptions:
        """Creates TrainOptions instance from cmdline arguments"""
        if args.f != "":
            jsonFile = Path(makePathAbsolute(args.f))
            if jsonFile.exists() and jsonFile.is_file():
                with jsonFile.open() as f:
                    content = f.read()
                    return jsonpickle.decode(content)
            else:
                raise ValueError("Could not find JSON input file")
        else:
            return cls(
                inputFile=args.i,
                outputDir=args.o,
                ecWeightsFile=args.ECmodel,
                model=args.model,
                target=args.target,
                fpSize=args.s,
                encFPSize=args.d,
                type=args.t,
                fpType=args.k,
            )


def parseInputPredict(parser: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    #    parser = argparse.ArgumentParser(description='Use models that have been generated by deepFPlearn-Train.py '
    #                                                 'tool to make predictions on chemicals (provide SMILES or '
    #                                                 'topological fingerprints).')
    parser.add_argument('-f', metavar='FILE', type=str,
                        help="Input JSON file that contains all information for predictions.",
                        required=False, default="")
    parser.add_argument('-i', metavar='FILE', type=str,
                        help="The file containing the data for the prediction. It is in"
                             "comma separated CSV format. The column named 'smiles' or 'fp'"
                             "contains the field to be predicted. Please adjust the type "
                             "that should be predicted (fp or smile) with -t option appropriately."
                             "An optional column 'id' is used to assign the outcomes to the"
                             "original identifiers. If this column is missing, the results are"
                             "numbered in the order of their appearance in the input file."
                             "A header is expected and respective column names are used.",
                        required=False)
    parser.add_argument('--ECmodel', metavar='FILE', type=str,
                        help='The encoder model weights. If provided the fingerprints are compressed prior '
                             'to prediction.',
                        required=False)
    parser.add_argument('--model', metavar='FILE', type=str,
                        help='The model weights of the feed forward network.',
                        required=False)
    parser.add_argument('--target', metavar='STR', type=str,
                        help='The name of the prediction target.',
                        required=False)
    parser.add_argument('-s', type=int,
                        help='Size of fingerprint that should be generated.',
                        default=2048)
    parser.add_argument('-d', metavar='INT', type=int,
                        help='Size of encoded fingerprint (z-layer of autoencoder).',
                        default=256)
    parser.add_argument('-o', metavar='FILE', type=str,
                        help='Output directory. It will contain logging information and an extended version of the '
                             'input file containing additional columns with the predictions of a random and the '
                             'trained model.')
    parser.add_argument('-t', metavar='STR', type=str, choices=['fp', 'smiles'],
                        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
                        default='smiles')
    parser.add_argument('-k', metavar='STR', type=str,
                        choices=['topological', 'MACCS'],  # , 'atompairs', 'torsions'],
                        help='The type of fingerprint to be generated/used in input file.',
                        default='topological')


def parseInputConvert(parser: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    parser.add_argument('-f', metavar='FILE', type=str,
                        help="Input directory where your CSV/TSV files are stored.",
                        required=True, default="")
