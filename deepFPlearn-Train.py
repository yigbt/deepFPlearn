import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')

# for fingerprint generation
from rdkit import DataStructs

# for NN model functions
from keras import optimizers

# import my own functions for deepFPlearn
import dfplmodule as dfpl

# ------------------------------------------------------------------------------------- #

def parseInput():
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    parser = argparse.ArgumentParser(description='Train a DNN to associate chemical fingerprints with a (set of)'
                                     'target(s). Trained models are saved to disk including fitted weights and '
                                     'can be used in the deepFPlearn-Predict.py tool to make predictions.')
    parser.add_argument('-i', metavar='FILE', type=str, nargs=1,
                        help="The file containin the data for training in (unquoted) "
                             "comma separated CSV format. First column contain the feature string in "
                             "form of a fingerprint or a SMILES (see -t option). "
                             "The remaining columns contain the outcome(s) (Y matrix). "
                             "A header is expected and respective column names are used "
                             "to refer to outcome(s) (target(s)).", required=True)
    parser.add_argument('-o', metavar='FILE', type=str, nargs=1,
                        help='Prefix of output file name. Trained model(s) and '
                             'respective stats will be returned in 2 output files '
                             'with this prefix. Default: prefix of input file name.')
    parser.add_argument('-t', metavar='STR', type=str, nargs=1, choices=['fp', 'smiles'],
                        help="Type of the chemical representation. Choices: 'fp', 'smiles'.",
                        required=True)
    parser.add_argument('-k', metavar='STR', type=str, nargs=1,
                        choices=['topological', 'MACCS'],  # , 'atompairs', 'torsions'],
                        help='The type of fingerprint to be generated/used in input file.',
                        default=['topological'])
    parser.add_argument('-e', metavar='INT', type=int, nargs=1,
                        help='Number of epochs that should be trained',
                        default=[50])

    return parser.parse_args()

# ------------------------------------------------------------------------------------- #

def trainNNmodels(model, modelfilepathprefix, pdx, y, split=0.8, e=50):
    """
    Train one model of the provided structure for each target (column) provided in y
    using the features x and the outcomes y and a train/validation data set split of
    split.
    :param model: a compiled neural network model
    :param x: a pandas data frame of training features, one row per data set, features in cols,
    rownames or numbers provided.
    :param y: a pandas data frame of training outcomes, one column per outcome.
    Cell value (0,1). Name of column used as name of target.
    :param split: the percentage data sets used for training. Remaining percentage is
    used in cross-validation steps.
    :param e: the number of epochs for training. Default: 50
    :param modelfilepathprefix: the path for saving the model weigths for each epoch
    :return: matrix of statistics per target
    """

    # for each target train a model
    stats = []

    # transform pd dataframe to numpy array for keras
    x = pdx.to_numpy()

    # learning rate
    lr = 0.001
    # type of optimizer
    adam = optimizers.Adam(lr=lr)

    for target in y.columns:
        # target=y.columns[2]
        tmp=y[target].astype('category')
        Y=np.asarray(tmp)
#        print(Y)
        naRows = np.isnan(Y)
        modelfilepathW=str(modelfilepathprefix) + '/model.' + target + '.weights.h5'
        modelfilepathM = str(modelfilepathprefix) + '/model.' + target + '.json'
        modelhistplotpathL = str(modelfilepathprefix) + '/model.' + target + '.loss.svg'
        modelhistplotpathA = str(modelfilepathprefix) + '/model.' + target + '.acc.svg'

        modelhistcsvpath=str(modelfilepathprefix) + '/model.' + target + '.history.csv'

        # define model structure - the same for all targets
        # An empty (weights) model needs to be defined each time prior to fitting
        model = dfpl.defineNNmodel(inputSize=pdx.shape[1])
        # compile model
        model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])

        # Train the model
        hist=model.fit(x[~naRows], Y[~naRows], epochs=e, validation_split=split,verbose=4)

        # serialize model to JSON
        model_json = model.to_json()
        with open(modelfilepathM, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(modelfilepathW)

        # print(hist.history)
        with open(modelhistcsvpath, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["metrictype", "epoch", "value"])
            for key, value in hist.history.items():
                for i, ep in enumerate(value):
                    writer.writerow([key, i+1, ep])

        # read it back in
        #with open('dict.csv') as csv_file:
         #   reader = csv.reader(csv_file)
          #  mydict = dict(reader)


        # plot accuracy and loss for the training and validation during training
        dfpl.plotTrainHistory(hist=hist, target=target, fileAccuracy=modelhistplotpathA, fileLoss=modelhistplotpathL)

        # plot weights


        scores=model.evaluate(x[~naRows],Y[~naRows],verbose=0)
        #model.save_weights(modelfilepath)

        stats.append([target, scores[0].__round__(2), scores[1].__round__(2)])
        print('\n' + target, "--> Loss:", scores[0].__round__(2), "Acc:", scores[1].__round__(2), sep=" ")

        print("FILES saved to disk for target " + target + ":")
        print("   [01] The trained model in JSON format: " + modelfilepathM)
        print("   [02] Weights of the model: " + modelfilepathW)
        print("   [03] Model history as .csv:" + modelhistcsvpath)
        print("   [04] Plot visualizing training accuracy: " + modelhistplotpathA)
        print("   [05] Plot visualizing training loss: " + modelhistplotpathL + '\n')
        print("The Path prefix for making predictions using deepFPlearn-predict is:\n")
        print(    str(modelfilepathprefix) + '/model.' + target)
        print('\n-------------------------------------------------------------------------------\n\n')

        del model
        del naRows
        del Y
        del tmp

    return stats


# ------------------------------------------------------------------------------------- #


def trainMultiNNmodel(model, x, y, split=0.8):
    """
    Train one multi-class model of the provided structure for all targets (columns) provided in y
    using the features x and the outcomes y and a train/validation data set split of
    split.
    :param model: a compiled neural network model
    :param x: a numpy array of training features, one row per data set, features in cols.
    :param y: a pandas data frame of training outcomes, one column per outcome.
    Cell value (0,1). Name of column used as name of target.
    :param split: the percentage data sets used for training. Remaining percentage is
    used in cross-validation steps.
    :return: matrix of statistics per target
    """

    stats = {}

    return stats


# ------------------------------------------------------------------------------------- #

def smilesSet2fpSet(csvfilename, outfilename, fptype):
    """

    :param csvfilename: csv file containing a column named 'smiles'
    :param fptype:
    :return: void
    """
    # csvfilename="/data/bioinf/projects/data/2019_Sun-etal_Supplement/results/05_04_dataKS.csv"
    # outfilename = "/data/bioinf/projects/data/2019_Sun-etal_Supplement/results/05_04_dataKS.fp.csv"

    # read csv and generate/add fingerprints to dict
    with open(csvfilename, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')

        feature = 'smiles'

        with open(outfilename, 'w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=['fp'])
            writer.writeheader()
            for row in reader:
                # smiles, need to be converted to fp first
                fp = smi2fp(smile=row[feature], fptype=fptype)
                writer.writerow({'fp':DataStructs.BitVectToText(fp)})

    return

# ===================================================================================== #


if __name__ == '__main__':

    # get all arguments
    args = parseInput()

    #print(args)
    #exit(1)

    # transform X to feature matrix
    # -i /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv
    # -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/
    # -t smiles -k topological -e 5
    #xmatrix = dfpl.XfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv", rtype="smiles", fptype="topological", printfp=True)
    xmatrix = dfpl.XfromInput(csvfilename=args.i[0], rtype=args.t[0], fptype=args.k[0], printfp=True)

    print(xmatrix.shape)

    # transform Y to feature matrix
    #ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv")
    ymatrix = dfpl.YfromInput(csvfilename=args.i[0])

    print(ymatrix.shape)

    # define model structure - the same for all targets
    model = dfpl.defineNNmodel(inputSize=xmatrix.shape[1])

    print(model.summary())

    # train one model per target (individually)
    #modelstats = trainNNmodels(model=model, modelfilepathprefix="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/", pdx=xmatrix, y=ymatrix, split=0.8, e=5)
    modelstats = trainNNmodels(model=model, modelfilepathprefix=args.o[0], pdx=xmatrix, y=ymatrix, split=0.8, e=args.e[0])

    print(modelstats)

    # generate stats

    # produce output
