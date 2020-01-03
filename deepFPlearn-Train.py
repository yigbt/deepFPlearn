import argparse
import csv
import numpy as np

# for fingerprint generation
from rdkit import DataStructs

# for NN model functions
from keras import optimizers

# import my own functions for deepFPlearn
import dfplmodule as dfpl
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

from time import time

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

def trainNNmodels(modelfilepathprefix, x, y, split=0.2, epochs=50):
    """
    Train individual models for all targets (columns) present in the provided target data (y).
    For each individual target the data is first subsetted to exclude NA values (for target associations).
    A random sample of the remaining data (size is the split fraction) is used for training and the
    remaining data for validation.

    :param modelfilepathprefix:
    :param x:
    :param y:
    :param split:
    :param epochs:
    :return:
    """

    # for testing
    #(modelfilepathprefix, x, y, split, epochs)  = (mfp, xmatrix, ymatrix, 0.2, 5)

    stats = []

    ### General model parameters
    # learning rate
    lr = 0.001
    # type of optimizer
    adam = optimizers.Adam(lr=lr)


    ### For each individual target
    for target in y.columns:
        # target=y.columns[0]
        modelfilepathW = str(modelfilepathprefix) + '/model.' + target + '.weights.h5'
        modelfilepathM = str(modelfilepathprefix) + '/model.' + target + '.json'
        modelhistplotpathL = str(modelfilepathprefix) + '/model.' + target + '.loss.svg'
        modelhistplotpathA = str(modelfilepathprefix) + '/model.' + target + '.acc.svg'
        modelhistcsvpath = str(modelfilepathprefix) + '/model.' + target + '.history.csv'
        outfilepath = str(modelfilepathprefix) + '/model.' + target + '.trainingResults.csv'

        # which rows contain 'NA' in target column
        tmp = y[target].astype('category')
        Y = np.asarray(tmp)
        naRows = np.isnan(Y)

        # transform pd dataframe to numpy array for keras
        X = x.to_numpy()

        # subset data according to target non-NA values
        Yt = Y[~naRows]
        Xt = X[~naRows]

        # randomly split into train and test sets
        (X_train, X_test, y_train, y_test) = train_test_split(Xt, Yt, test_size=split, random_state=0)

        # define model structure - the same for all targets
        # An empty (weights) model needs to be defined each time prior to fitting
        #model = dfpl.defineNNmodel(inputSize=X_train.shape[1])
        # compile model
        #model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])

        # standard parameters are the tuning results
        model = dfpl.defineNNmodel2()

        start = time()

        # train and validate
        hist = model.fit(X_train, y_train, epochs=epochs, verbose=2, validation_split=0.2,
                         validation_data=(X_test, y_test), batch_size=128) # this overwrites val_split!


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
        # svmtest

        predictions = model.predict(X_test)

        confusion_matrix(y_test, predictions.round(), normalize='all')

        scores=model.evaluate(X_test, y_test,verbose=0)

        stats.append([target, scores[0].__round__(2), scores[1].__round__(2)])

        # write stats to file
        file = open(outfilepath, "a")

        print('\n' + target, "--> Loss:", scores[0].__round__(2), "Acc:", scores[1].__round__(2), sep=" ")
        file.write("\n" + target + " -----------------------------------------------------------------------------\n")
        file.write("Loss: " + scores[0].__round__(2) + "\n")
        file.write("Acc:  " + scores[1].__round__(2) + "\n\n")
        
        file.write("Training time: %s min\n\n" % str(round((time() - start) / 60, ndigits=2)))

        file.write("FILES saved to disk for target " + target + ":")
        file.write("   [01] The trained model in JSON format: " + modelfilepathM)
        file.write("   [02] Weights of the model: " + modelfilepathW)
        file.write("   [03] Model history as .csv:" + modelhistcsvpath)
        file.write("   [04] Plot visualizing training accuracy: " + modelhistplotpathA)
        file.write("   [05] Plot visualizing training loss: " + modelhistplotpathL + '\n')
        file.write("The Path prefix for making predictions using deepFPlearn-predict is:\n")
        file.write(    str(modelfilepathprefix) + '/model.' + target)
        file.write('\n-------------------------------------------------------------------------------\n\n')
        ### find best performing parameters
        file.close()

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

    # increase recursion limit to draw heatmaps later
    sys.setrecursionlimit(10000)

    # get all arguments
    args = parseInput()

    #print(args)
    #exit(1)

    # transform X to feature matrix
    # -i /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv
    # -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/
    # -t smiles -k topological -e 5
    #xmatrix = dfpl.XfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv", rtype="smiles", fptype="topological", printfp=False)
    #xmatrix = dfpl.XfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/toxCastData/AhR/results/02_training_Ahr.noNA.csv", rtype="smiles", fptype="topological", printfp=True)
    xmatrix = dfpl.XfromInput(csvfilename=args.i[0], rtype=args.t[0], fptype=args.k[0], printfp=True)

    print(xmatrix.shape)

    # transform Y to feature matrix
    #ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv")
    #ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/toxCastData/AhR/results/02_training_Ahr.noNA.csv")
    ymatrix = dfpl.YfromInput(csvfilename=args.i[0])

    print(ymatrix.shape)

    # define model structure - the same for all targets
#    model = dfpl.defineNNmodel(inputSize=xmatrix.shape[1])

#    print(model.summary())

    epochs = args.e[0] # epochs=20
    mfp = args.o[0] # mfp = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/"#AhR"
    # train one model per target (individually)
    #modelstats = trainNNmodels(model=model, modelfilepathprefix="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/AhR", pdx=xmatrix, y=ymatrix, split=0.8, e=args.e[0], valdata=(xmatrixTest, ymatrixTest))
    #modelstats = trainNNmodels(model=model, modelfilepathprefix=mfp, pdx=xmatrix, y=ymatrix, split=0.8, e=epochs, valdata=(xmatrixTest, ymatrixTest))
    modelstats = trainNNmodels(modelfilepathprefix=mfp, x=xmatrix, y=ymatrix, split=0.8, epochs=epochs)

    print(modelstats)

    # generate stats

    # produce output
