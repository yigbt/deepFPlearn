import argparse
import csv
import pandas as pd
import numpy as np

# for fingerprint generation
from rdkit import DataStructs

# import my own functions for deepFPlearn
import dfplmodule as dfpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc

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
    parser.add_argument('-i', metavar='FILE', type=str,
                        help="The file containin the data for training in (unquoted) "
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
                        required=True)
    parser.add_argument('-k', metavar='STR', type=str,
                        choices=['topological', 'MACCS'],  # , 'atompairs', 'torsions'],
                        help='The type of fingerprint to be generated/used in input file.',
                        default=['topological'])
    parser.add_argument('-s', type=int,
                        help = 'Size of fingerprint that should be generated.',
                        default=2048)
    parser.add_argument('-a', action='store_true',
                        help='Use autoencoder to reduce dimensionality of fingerprint. Default: not set.')
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

    return parser.parse_args()

# ------------------------------------------------------------------------------------- #

def defineOutfileNames(pathprefix, mtype, target):
    """
    This function returns the required paths for output files or directories.

    :param pathprefix: A file path prefix for all files.
    :param mtype: The model type. Its set by the trainNNmodels function with information on autoencoder or not,
    and if AC is used, then with its parameters.
    :param target: The name of the target.

    :return: A tuple of 14 output file names.
    """

    modelfilepathW = str(pathprefix) + '/model.' + mtype + '.' + target + '.weights.h5'
    modelfilepathM = str(pathprefix) + '/model.' + mtype + '.' + target + '.json'
    modelhistplotpathL = str(pathprefix) + '/model.' + mtype + '.' + target + '.loss.svg'
    modelhistplotpathA = str(pathprefix) + '/model.' + mtype + '.' + target + '.acc.svg'
    modelhistplotpath = str(pathprefix) + '/model.' + mtype + '.' + target + '.history.svg'
    modelhistcsvpath = str(pathprefix) + '/model.' + mtype + '.' + target + '.history.csv'
    modelvalidation = str(pathprefix) + '/model.' + mtype + '.' + target + '.validation.csv'
    modelAUCfile = str(pathprefix) + '/model.' + mtype + '.' + target + '.auc.svg'
    modelAUCfiledata = str(pathprefix) + '/model.' + mtype + '.' + target + '.auc.data.csv'
    outfilepath = str(pathprefix) + '/model.' + mtype + '.' + target + '.trainingResults.txt'
    checkpointpath = str(pathprefix) + '/model.' + mtype + '.' + target + '.checkpoint.model.hdf5'
    checkpointpathAC = str(pathprefix) + '/model.' + mtype + '.' + target + '.checkpoint.AC-model.hdf5'
    modelheatmapX = str(pathprefix) + '/model.' + mtype + '.' + target + '.heatmap.X.svg'
    modelheatmapZ = str(pathprefix) + '/model.' + mtype + '.' + target + '.AC.heatmap.Z.svg'

    return (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
            modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
            modelAUCfiledata, outfilepath, checkpointpath, checkpointpathAC,
            modelheatmapX, modelheatmapZ)

# ------------------------------------------------------------------------------------- #

def eval01Distributions(Xt, Yt, y_train, y_test, verbosity=0):
    """
    Evaluate the percentage of 0 values in the outcome variable of the whole dataset and the splitted (train,test)
    dataset, and the percentage of 0 values in the feature matrix.

    :param Xt: The whole feature matrix
    :param Yt: The whole outcome vector
    :param y_train: The outcome vector of the training set
    :param y_test: The outcome vector of the test set
    :param verbosity: The verbosity level. Info is only printed if verbosity is not 0.

    :return: Nothing is returned.
    """

    if verbosity == 0:
        return
    else:
        unique, counts = np.unique(Yt, return_counts=True)
        perc = round(100 / len(Yt) * counts[1])
        print(f"[INFO:] Percentage of '1' values in outcome variable (whole dataset): {perc}\n")

        uniqueRtr, countsRtr = np.unique(y_train, return_counts=True)
        uniqueRte, countsRte = np.unique(y_test, return_counts=True)
        perc = round(100 / len(y_train) * countsRtr[1])
        print(f"[INFO:] Percentage of '1' values in training outcomes: {perc}\n")
        perc = round(100 / len(y_test) * countsRte[1])
        print(f"[INFO:] Percentage of '1' values in test outcomes: {perc}\n")

        print(
            f"[INFO:] Percentage of '0' values in fingerprints: {round(np.sum(Xt == 0) / (np.sum(Xt == 0) + np.sum(Xt == 1)), ndigits=4)}")
    return

# ------------------------------------------------------------------------------------- #

def trainNNmodels(modelfilepathprefix, x, y, split=0.2, epochs=50, params=None, enc_dim=256, autoenc=False, verbose=0):
    """
    Train individual models for all targets (columns) present in the provided target data (y).
    For each individual target the data is first subsetted to exclude NA values (for target associations).
    A random sample of the remaining data (size is the split fraction) is used for training and the
    remaining data for validation.

    :param modelfilepathprefix: A path prefix for all output files
    :param x: The feature matrix.
    :param y: The outcome matrix.    :param split: The percentage of data used for validation.
    :param epochs: The number of epochs for training the autoencoder and the DNN for classification.
    Note: Early stopping is enabled.
    :param params: A .csv files containing paramters that should be evaluated. See file tunedParams.csv.
    :param enc_dim: The dimension of bottle neck layer (z) of the autoencoder.
    :param autoenc: Use the autoencoder.
    :param verbose: Verbosity level.

    :return: A list with loss and accuracy values for each individual model.
    """

    size = x.shape[1]

    stats = []

    if params:
        parameters = pd.read_csv(params)

    # add a target 'ED' for association to ANY of the target genes
    # maybe it improves the detection of '1's due to increased data set
    mysum = y.sum(axis=1)
    y['ED'] = [0 if s == 0 else 1 for s in mysum]

    ### For each individual target
    for target in y.columns:
        # target=y.columns[0] # --> only for testing the code

        if autoenc:
            modeltype = str(size) + '.' + str(enc_dim)
        else:
            modeltype = str(size) + '.noAC'

        # define all the output file/path names
        (modelfilepathW, modelfilepathM, modelhistplotpathL, modelhistplotpathA,
         modelhistplotpath, modelhistcsvpath, modelvalidation, modelAUCfile,
         modelAUCfiledata, outfilepath, checkpointpath, checkpointpathAC,
         modelheatmapX, modelheatmapZ) = defineOutfileNames(pathprefix=modelfilepathprefix, mtype=modeltype, target=target)

        # which rows contain 'NA' in target column
        tmp = y[target].astype('category')
        Y = np.asarray(tmp)
        naRows = np.isnan(Y)

        # transform pd dataframe to numpy array for keras
        X = x.to_numpy()

        # subset data according to target non-NA values
        Ytall = Y[~naRows]
        Xtall = X[~naRows]

        # remove all duplicated feature values - outcome pairs
        (Xt, Yt) = dfpl.removeDuplicates(x=Xtall, y=Ytall)

        # split data (features and outcome) into test and train set
        # Note: the distribution of outcome values is retained
        (X_train, X_test, y_train, y_test) = train_test_split(Xt, Yt, test_size=(1-split), random_state=0)

        eval01Distributions(Xt=Xt, Yt=Yt, y_train=y_train, y_test=y_test, verbosity=verbose)

        # Plot a heatmap of the 0-1 values of the feature matrix used for training
        # This should contain mainly zeros for this problem
        dfpl.plotHeatmap(X_train, filename=modelheatmapX, title=("X representation " + target))

        # Train the autoencoder (AC) to reduce the feature vector for the input layer of the DNN classificator
        if autoenc:
            (Z_train, Z_test) = dfpl.trainAutoencoder(checkpointpath=checkpointpathAC,
                                                      X_train=X_train, X_test=X_test,
                                                      y_train=y_train, y_test=y_train,
                                                      epochs=epochs)

            dfpl.plotHeatmap(Z_train, filename=modelheatmapZ,title=("Z representation "+target))

        else:
            (Z_train, Z_test) = (X_train, X_test)

        # standard parameters are the tuning results
#        model = dfpl.defineNNmodel2()
        if params:
            ps = parameters.loc[parameters['target'] == target]
            model = dfpl.defineNNmodel(inputSize=X_train.shape[1], activation=ps['activation'][0], optimizer=ps['optimizer'][0])

            start = time()

            callback_list = dfpl.defineCallbacks(checkpointpath=checkpointpath, patience=50,
                                                 rlrop=True, rlropfactor=0.1, rlroppatience=100)

            # train and validate
            hist = model.fit(X_train, y_train,
                             callbacks=callback_list,
                             epochs=ps['epochs'][0],
                             batch_size=ps['batch_size'][0], verbose=2,#,
                             #validation_split=0.2,
                             validation_data=(X_test, y_test)) # this overwrites val_split!
            trainTime = str(round((time() - start) / 60, ndigits=2))
        else:
            model = dfpl.defineNNmodel(inputSize=Z_train.shape[1]) #X_train.shape[1])

            callback_list = dfpl.defineCallbacks(checkpointpath=checkpointpath, patience=50,
                                                 rlrop=True, rlropfactor=0.1, rlroppatience=100)
            # measure the training time
            start = time()

            # train and validate
            hist = model.fit(Z_train, y_train, #X_train, y_train,
                             callbacks=callback_list,
                             epochs=epochs, batch_size=128, verbose=2, # validation_split=0.2,
                             validation_data=(Z_test, y_test)) # this overwrites val_split!

            trainTime = str(round((time() - start) / 60, ndigits=2))
            if verbose > 0:
                print(f"[INFO:] Computation time for training the classification DNN: {trainTime}")

        dfpl.plot_history(history=hist, file=modelhistplotpath)
        histDF = pd.DataFrame(hist.history)
        histDF.to_csv(modelhistcsvpath)

        # plot accuracy and loss for the training and validation during training
        dfpl.plotTrainHistory(hist=hist, target=target, fileAccuracy=modelhistplotpathA, fileLoss=modelhistplotpathL)

        # load checkpoint model with min(val_loss)
        trainedmodel = dfpl.defineNNmodel(inputSize=Z_train.shape[1])

        predictions_random = trainedmodel.predict(Z_test)

        trainedmodel.load_weights(checkpointpath)

        predictions = trainedmodel.predict(Z_test)

        # save validation data to .csv file
        validation = pd.DataFrame({'predicted': predictions.ravel(),
                                   'true': list(y_test),
                                   'predicted_random':predictions_random.ravel(),
                                   'modeltype': modeltype})
        validation.to_csv(modelvalidation)

        # generate the AUC-ROC curve data from the validation data
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, predictions, drop_intermediate=False)
        auc_keras = auc(fpr_keras, tpr_keras)

        aucdata = pd.DataFrame(list(zip(fpr_keras,
                                        tpr_keras,
                                        [auc_keras for x in range(1,len(fpr_keras))],
                                        [target for x in range(1,len(fpr_keras))])),
                               columns=['fpr', 'tpr', 'auc', 'target'])
        aucdata.to_csv(modelAUCfiledata)

        dfpl.plotAUC(fpr=fpr_keras, tpr=tpr_keras, target=target, auc=auc_keras, filename=modelAUCfile)

        print(f'CFM: \n{confusion_matrix(predictions.round(), y_test)}')

        scores=trainedmodel.evaluate(Z_test, y_test, verbose=0)

        print(f'TARGET: {target} Loss: {scores[0].__round__(2)} Acc: {scores[1].__round__(2)}')

        stats.append([target, scores[0].__round__(2), scores[1].__round__(2)])

        # write stats to file
        file = open(outfilepath, "a")

        file.write("\n" + target + " -----------------------------------------------------------------------------\n")
        file.write("Loss: " + str(scores[0].__round__(2)) + "\n")
        file.write("Acc:  " + str(scores[1].__round__(2)) + "\n\n")
        
        file.write("Training time: %s min\n\n" % trainTime)

        file.write("FILES saved to disk for target " + target + ":")
        file.write("   [01] The trained model in JSON format: " + modelfilepathM)
        file.write("   [02] Weights of the model: " + modelfilepathW)
        file.write("   [03] Model history as .csv:" + modelhistcsvpath)
        file.write("   [04] Plot visualizing training accuracy: " + modelhistplotpathA)
        file.write("   [05] Plot visualizing training loss: " + modelhistplotpathL + '\n')
        file.write("The Path prefix for making predictions using deepFPlearn-predict is:\n")
        file.write(    str(modelfilepathprefix) + '/model.' + target)
        file.write('\n-------------------------------------------------------------------------------\n\n')
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

# [['AR', 0.15, 0.86], ['ER', 0.18, 0.81], ['GR', 0.09, 0.92], ['Aromatase', 0.1, 0.91], ['TR', 0.08, 0.92], ['PPARg', 0.12, 0.88], ['ED', 0.21, 0.78]]
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
    #sys.setrecursionlimit(10000)

    # get all arguments
    args = parseInput()

    #print(args)
    #exit(1)

    # transform X to feature matrix
    # -i /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv
    # -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/
    # -t smiles -k topological -e 5
    #xmatrix = dfpl.XfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv", rtype="smiles", fptype="topological", printfp=False)
    # xmatrix = dfpl.XfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.999.csv", rtype="smiles", fptype="topological", printfp=True, size=999)
    #xmatrix = dfpl.XfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/toxCastData/AhR/results/02_training_Ahr.noNA.csv", rtype="smiles", fptype="topological", printfp=True)
    xmatrix = dfpl.XfromInput(csvfilename=args.i, rtype=args.t, fptype=args.k, printfp=True, size=args.s)

    print(xmatrix.shape)

    # transform Y to feature matrix
    #ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv")
    #ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.999.csv")
    #ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/toxCastData/AhR/results/02_training_Ahr.noNA.csv")
    ymatrix = dfpl.YfromInput(csvfilename=args.i)

    print(ymatrix.shape)

    # define model structure - the same for all targets
#    model = dfpl.defineNNmodel(inputSize=xmatrix.shape[1])

#    print(model.summary())

    epochs = args.e # epochs=20
    mfp = args.o # mfp = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/"#AhR"
    # mfp = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/2020-01-27_11:28:00/"
    # train one model per target (individually)
    #modelstats = trainNNmodels(model=model, modelfilepathprefix="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/AhR", pdx=xmatrix, y=ymatrix, split=0.8, e=args.e[0], valdata=(xmatrixTest, ymatrixTest))
    #modelstats = trainNNmodels(model=model, modelfilepathprefix=mfp, pdx=xmatrix, y=ymatrix, split=0.8, e=epochs, valdata=(xmatrixTest, ymatrixTest))
    # (modelfilepathprefix, x, y, split, epochs) = (mfp, xmatrix, ymatrix, 0.8, 1000)
    if args.p:
        modelstats = trainNNmodels(modelfilepathprefix=mfp, x=xmatrix, y=ymatrix, split=0.8, params=args.p)
    else:
        modelstats = trainNNmodels(modelfilepathprefix=args.o, x=xmatrix, y=ymatrix, split=0.8, enc_dim=args.d, epochs=args.e, autoenc=args.a)

    print(modelstats)

    # generate stats

    # produce output

# xmatrix = dfpl.XfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv", rtype="smiles", fptype="topological", size=2048, printfp=False)
# ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv")
# mfp = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/2020-01-27_11:28:00/"
# (modelfilepathprefix, x, y, split, epochs) = (mfp, xmatrix, ymatrix, 0.8, 200)
