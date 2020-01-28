import argparse
import csv
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from numpy import argmax
import matplotlib.pyplot as plt

# for fingerprint generation
from rdkit import DataStructs

# for NN model functions
from keras.utils import to_categorical
from keras import optimizers
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json

# import my own functions for deepFPlearn
import dfplmodule as dfpl
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
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
    parser.add_argument('-s', metavar='INT', type=int, nargs=1,
                        help = 'Size of fingerprint that should be generated.',
                        default=2048)
    parser.add_argument('-d', metavar='INT', type=int, nargs=1,
                        help='Size of encoded fingerprint (z-layer of autoencoder).',
                        default=256)
    parser.add_argument('-e', metavar='INT', type=int, nargs=1,
                        help='Number of epochs that should be trained',
                        default=[50])
    parser.add_argument('-p', metavar='FILE', type=str, nargs=1,
                        help="CSV file containing the parameters for the epochs per target model."
                             "The target abbreviation should be the same as in the input file and"
                             "the columns/parameters are: \n\ttarget,batch_size,epochs,optimizer,activation."
                             "Note that values in from file overwrite -e option!")

    return parser.parse_args()

# ------------------------------------------------------------------------------------- #

def step_decay(history, losses):
    if float(2*np.sqrt(np.array(history.losses[-1])))<0.3:
        lrate=0.01*1/(1+0.1*len(history.losses))
        momentum=0.8
        decay_rate=2e-6
        return lrate
    else:
        lrate=0.1
        return lrate

# ------------------------------------------------------------------------------------- #

def trainNNmodels(modelfilepathprefix, x, y, split=0.2, epochs=50, params=None, enc_dim=256):
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
    :param params: File containing the parameters per model
    :return:
    """

    # for testing
    #(modelfilepathprefix, x, y, split, epochs)  = (mfp, xmatrix, ymatrix, 0.2, 5)

    size = x.shape[1]

    stats = []

    #params="/home/hertelj/git-hertelj/code/2019_deepFPlearn/tunedParams.csv"

    if params:
        parameters = pd.read_csv(params)

    # add a target 'ED' for association to ANY of the target genes
    # maybe it improves the detection of '1's due to increased data set
    mysum = y.sum(axis=1)
    y['ED'] = [0 if s == 0 else 1 for s in mysum]

    ### For each individual target
    for target in y.columns:
        # target=y.columns[0]
        modelfilepathW = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.weights.h5'
        modelfilepathM = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.json'
        modelhistplotpathL = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.loss.svg'
        modelhistplotpathA = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.acc.svg'
        modelhistplotpath = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.history.svg'
        modelhistcsvpath = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.history.csv'
        modelvalidation = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.validation.csv'
        modelAUCfile = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.auc.svg'
        modelAUCfiledata = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.auc.data.csv'
        outfilepath = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.trainingResults.txt'
        checkpointpath = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.checkpoint.model.hdf5'
        checkpointpathAC = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.checkpoint.AC-model.hdf5'
        modelheatmapX = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.AC.heatmap.X.svg'
        modelheatmapZ = str(modelfilepathprefix) + '/model.' + size + '.' + enc_dim + '.' + target + '.AC.heatmap.Z.svg'

        # which rows contain 'NA' in target column
        tmp = y[target].astype('category')
        Y = np.asarray(tmp)
        naRows = np.isnan(Y)

        # transform pd dataframe to numpy array for keras
        X = x.to_numpy()

        # subset data according to target non-NA values
        Yt = Y[~naRows]
        Xt = X[~naRows]

        unique, counts = np.unique(Yt, return_counts=True)
        perc = round(100 / len(Yt) * counts[1])
        print(f"Percentage of '1' values in whole dataset: {perc}\n")
        (X_train, X_test, y_train, y_test) = train_test_split(Xt, Yt, test_size=(1-split), random_state=0)
        uniqueRtr, countsRtr = np.unique(y_train, return_counts=True)
        uniqueRte, countsRte = np.unique(y_test, return_counts=True)
        perc = round(100/len(y_train)*countsRtr[1])
        print(f"Percentage of '1' values in training set: {perc}\n")
        perc = round(100/len(y_test)*countsRte[1])
        print(f"Percentage of '1' values in test set: {perc}\n")
        # define model structure - the same for all targets
        # An empty (weights) model needs to be defined each time prior to fitting
        #model = dfpl.defineNNmodel(inputSize=X_train.shape[1])
        # compile model
        #model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])

        print(f"Percentage of '0' values in fingerprints: {round(np.sum(Xt==0)/(np.sum(Xt==0)+np.sum(Xt==1)), ndigits=4)}")

        dfpl.plotHeatmap(X_train, filename=modelheatmapX, title=("X representation " + target))

        checkpoint = ModelCheckpoint(checkpointpathAC, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min', save_weights_only=True)

        # enable early stopping if val_loss is not improving anymore
        earlystop = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=20,
                                  verbose=1,
                                  restore_best_weights=True)

        callback_list = [checkpoint, earlystop]

        # use the autoencoder to reduce the feature set
        # (autoencoder, encoder) = autoencoderModel(input_size=2048)
        (autoencoder, encoder) = dfpl.autoencoderModel(input_size=X_train.shape[1], encoding_dim=enc_dim)

        autohist = autoencoder.fit(X_train, X_train,
                            callbacks=callback_list,
                            epochs=1000, batch_size=128, shuffle=True, verbose=2,
                            validation_data=(X_test, X_test))

        # model needs to be saved and restored when predicting new input!
        # use encode() of train data as input for DL model to associate to chemical
        Z_train = encoder.predict(X_train)
        Z_test = encoder.predict(X_test)

        dfpl.plotHeatmap(Z_train, filename=modelheatmapZ,title=("Z representation "+target))

        # standard parameters are the tuning results
#        model = dfpl.defineNNmodel2()
        if params:
            ps = parameters.loc[parameters['target'] == target]
            model = dfpl.defineNNmodel(inputSize=X_train.shape[1], activation=ps['activation'][0], optimizer=ps['optimizer'][0])

            start = time()
            # define the checkpoint
            checkpoint = ModelCheckpoint(checkpointpath, monitor='val_loss', verbose=1,
                                         save_best_only=True, mode='min')

            # enable early stopping if val_loss is not improving anymore
            earlystop = EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=20,
                                      verbose=1,
                                      restore_best_weights=True)
            # schedule learning rate based on current epoch
            #history=Lo
            #scheduler = LearningRateScheduler(stop_decay, verbose=0)  # schedule is a function

            callback_list = [checkpoint, earlystop]#, scheduler]

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

            # define the checkpoint
            checkpoint = ModelCheckpoint(checkpointpath, monitor='val_loss', verbose=1,
                                         save_best_only=True, mode='min', save_weights_only=True)

            # enable early stopping if val_loss is not improving anymore
            # reduce overfitting by adding an early stopping to an existing model.
            earlystop = EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=50,
                                      verbose=1,
                                      restore_best_weights=True)

            rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)

            callback_list = [checkpoint, earlystop, rlrop]  # , scheduler]

            start = time()
            # train and validate
            hist = model.fit(Z_train, y_train, #X_train, y_train,
                             callbacks=callback_list,
                             epochs=epochs, batch_size=128, verbose=2, # validation_split=0.2,
                             validation_data=(Z_test, y_test)) # this overwrites val_split!
            trainTime = str(round((time() - start) / 60, ndigits=2))

        dfpl.plot_history(history=hist, file=modelhistplotpath)
        histDF = pd.DataFrame(hist.history)
        histDF.to_csv(modelhistcsvpath)

        # serialize model to JSON
        # model_json = model.to_json()
        # with open(modelfilepathM, "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # model.save_weights(modelfilepathW)
        #
        # # print(hist.history)
        # with open(modelhistcsvpath, 'w') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerow(["metrictype", "epoch", "value"])
        #     for key, value in hist.history.items():
        #         for i, ep in enumerate(value):
        #             writer.writerow([key, i+1, ep])

        # read it back in
        #with open('dict.csv') as csv_file:
         #   reader = csv.reader(csv_file)
          #  mydict = dict(reader)


        # plot accuracy and loss for the training and validation during training
        dfpl.plotTrainHistory(hist=hist, target=target, fileAccuracy=modelhistplotpathA, fileLoss=modelhistplotpathL)

        # plot weights
        # svmtest

        # use best saved model for prediction of validation data
        #new_model = load_model(checkpointpath)
        #assert_allclose(model.predict(x_train),
        #                new_model.predict(x_train),
        #                1e-5)

        # load checkpoint model with min(val_loss)
        trainedmodel = dfpl.defineNNmodel(inputSize=Z_train.shape[1])#X_train.shape[1])

        predictions_random = trainedmodel.predict(Z_test)#X_test)

        trainedmodel.load_weights(checkpointpath)

        predictions = trainedmodel.predict(Z_test)#X_test)

        validation = pd.DataFrame({'predicted': predictions.ravel(), 'true': list(y_test), 'predicted_random':predictions_random.ravel()})
        validation.to_csv(modelvalidation)


        #fpr_keras, tpr_keras, thresholds_keras = roc_curve(argmax(y_test, axis=1), argmax(predictions, axis=1).round(), drop_intermediate=False)
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, predictions, drop_intermediate=False)
        auc_keras = auc(fpr_keras, tpr_keras)

        aucdata = pd.DataFrame(list(zip(fpr_keras,
                                        tpr_keras,
                                        [auc_keras for x in range(1,len(fpr_keras))],
                                        [target for x in range(1,len(fpr_keras))])),
                               columns=['fpr', 'tpr', 'auc', 'target'])
        aucdata.to_csv(modelAUCfiledata)

        dfpl.plotAUC(fpr=fpr_keras, tpr=tpr_keras, target=target, auc=auc_keras, filename=modelAUCfile)
        #
        # plt.figure(3)
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        # plt.xlabel('False positive rate')
        # plt.ylabel('True positive rate')
        # plt.title(f'ROC curve {target}')
        # plt.legend(loc='best')
        # #plt.show()
        # plt.savefig(fname=modelAUCfile, format='svg')

        print(f'CFM: \n{confusion_matrix(predictions.round(), y_test)}')

        scores=trainedmodel.evaluate(Z_test, y_test, verbose=0)#X_test, y_test,verbose=0)

        print(f'TARGET: {target} Loss: {scores[0].__round__(2)} Acc: {scores[1].__round__(2)}')

        stats.append([target, scores[0].__round__(2), scores[1].__round__(2)])

        # write stats to file
        file = open(outfilepath, "a")

        #print('\n' + target, "--> Loss:", scores[0].__round__(2), "Acc:", scores[1].__round__(2), sep=" ")
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
    xmatrix = dfpl.XfromInput(csvfilename=args.i[0], rtype=args.t[0], fptype=args.k[0], printfp=True, size=args.s[0])

    print(xmatrix.shape)

    # transform Y to feature matrix
    #ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv")
    #ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.999.csv")
    #ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/toxCastData/AhR/results/02_training_Ahr.noNA.csv")
    ymatrix = dfpl.YfromInput(csvfilename=args.i[0])

    print(ymatrix.shape)

    # define model structure - the same for all targets
#    model = dfpl.defineNNmodel(inputSize=xmatrix.shape[1])

#    print(model.summary())

    epochs = args.e[0] # epochs=20
    mfp = args.o[0] # mfp = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/"#AhR"
    # mfp = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/2020-01-27_11:28:00/"
    # train one model per target (individually)
    #modelstats = trainNNmodels(model=model, modelfilepathprefix="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/AhR", pdx=xmatrix, y=ymatrix, split=0.8, e=args.e[0], valdata=(xmatrixTest, ymatrixTest))
    #modelstats = trainNNmodels(model=model, modelfilepathprefix=mfp, pdx=xmatrix, y=ymatrix, split=0.8, e=epochs, valdata=(xmatrixTest, ymatrixTest))
    # (modelfilepathprefix, x, y, split, epochs) = (mfp, xmatrix, ymatrix, 0.8, 1000)
    if args.p:
        modelstats = trainNNmodels(modelfilepathprefix=mfp, x=xmatrix, y=ymatrix, split=0.8, params=args.p[0])
    else:
        modelstats = trainNNmodels(modelfilepathprefix=mfp, x=xmatrix, y=ymatrix, split=0.8, enc_dim=args.d[0], epochs=epochs)

    print(modelstats)

    # generate stats

    # produce output

# xmatrix = dfpl.XfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv", rtype="smiles", fptype="topological", size=2048, printfp=False)
# ymatrix = dfpl.YfromInput(csvfilename="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv")
# mfp = "/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/2020-01-27_11:28:00/"
# (modelfilepathprefix, x, y, split, epochs) = (mfp, xmatrix, ymatrix, 0.8, 200)
