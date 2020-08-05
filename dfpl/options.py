from __future__ import annotations
from dataclasses import dataclass
import argparse
from pathlib import Path


@dataclass
class TrainOptions:
    inputFile: str = ""
    outputDir: str = "modeltraining/"
    acFile: str = "modeltraining/Sun_etal_dataset.AC.encoder.weights.hdf5"
    type: str = "smiles"
    fpType: str = "topological"  # also "MACCS", "atompairs"
    epochs: int = 512
    fpSize: int = 2048
    encFPSize: int = 256
    kFolds: int = 5
    testingFraction: float = 0.2
    enableMultiLabel: bool = False
    verbose: int = 0
    trainAC: bool = True    # if set to False, an AC weight file must be provided!
    trainFNN: bool = True

    @classmethod
    def fromCmdArgs(cls, args: argparse.Namespace) -> TrainOptions:
        """Creates TrainOptions instance from cmdline arguments"""
        inst = cls(
            inputFile=args.i,
            outputDir=args.o,
            acFile=args.a,
            type=args.t,
            fpType=args.k,
            fpSize=args.s,
            encFPSize=args.d,
            epochs=args.e,
            kFolds=args.K,
            testingFraction=args.l,
            enableMultiLabel=args.m,
            verbose=args.v,
            trainAC=args.trainAC,
            trainFNN=args.trainFNN
        )
        # While we're at it, check if the output dir exists and if not, create it
        out = Path(inst.outputDir)
        if not out.is_dir() and not out.is_file():
            out.mkdir(out)
        return inst


def createCommandlineParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='deepFPlearn')
    subparsers = parser.add_subparsers(help="Sub programs of deepFPlearn")

    parser_train = subparsers.add_parser("train", help="Train new models with your data")
    parser_train.set_defaults(method="train")
    parseInputTrain(parser_train)

    parser_predict = subparsers.add_parser("predict", help="Predict your data with existing models")
    parser_predict.set_defaults(method="predict")
    parseInputPredict(parser_predict)
    return parser


def parseInputTrain(parser: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    #    parser = argparse.ArgumentParser(description='Train a DNN to associate chemical fingerprints with a (set of)'
    #                                     'target(s). Trained models are saved to disk including fitted weights and '
    #                                     'can be used in the deepFPlearn-Predict.py tool to make predictions.')
    parser.add_argument('-i', metavar='FILE', type=str,
                        help="The file containing the data for training in (unquoted) "
                             "comma separated CSV format. First column contain the feature string in "
                             "form of a fingerprint or a SMILES (see -t option). "
                             "The remaining columns contain the outcome(s) (Y matrix). "
                             "A header is expected and respective column names are used "
                             "to refer to outcome(s) (target(s)).", required=True)
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
    parser.add_argument('-a', type=str, metavar='FILE', default=None,
                        help='The .hdf5 file of a trained autoencoder (e.g. from a previous'
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




def parseInputPredict(parser: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    #    parser = argparse.ArgumentParser(description='Use models that have been generated by deepFPlearn-Train.py '
    #                                                 'tool to make predictions on chemicals (provide SMILES or '
    #                                                 'topological fingerprints).')
    parser.add_argument('-i', metavar='FILE', type=str,
                        help="The file containin the data for the prediction. It is in"
                             "comma separated CSV format. The column named 'smiles' or 'fp'"
                             "contains the field to be predicted. Please adjust the type "
                             "that should be predicted (fp or smile) with -t option appropriately."
                             "An optional column 'id' is used to assign the outcomes to the"
                             "original identifieres. If this column is missing, the results are"
                             "numbered in the order of their appearance in the input file."
                             "A header is expected and respective column names are used.",
                        required=True)
    parser.add_argument('--ACmodel', metavar='FILE', type=str,
                        help='The autoencoder model weights',
                        required=True)
    parser.add_argument('--model', metavar='FILE', type=str,
                        help='The predictor model weights',
                        required=True)
    parser.add_argument('-o', metavar='FILE', type=str,
                        help='Output file name. It containes a comma separated list of '
                             "predictions for each input row, for all targets. If the file 'id'"
                             "was given in the input, respective IDs are used, otherwise the"
                             "rows of output are numbered and provided in the order of occurence"
                             "in the input file.")
    parser.add_argument('-t', metavar='STR', type=str, choices=['fp', 'smiles'],
                        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
                        default='smiles')
    parser.add_argument('-k', metavar='STR', type=str,
                        choices=['topological', 'MACCS'],  # , 'atompairs', 'torsions'],
                        help='The type of fingerprint to be generated/used in input file.',
                        default='topological')
