import argparse
from argparse import Namespace
import pandas as pd
import numpy as np

import logging

import fingerprint as fp
import dfplmodule as dfpl

import importlib
importlib.reload(fp)
importlib.reload(dfpl)

# args = Namespace(i='/data/bioinf/projects/data/2020_deepFPlearn/dataSources/Sun_et_al/Sun_etal_dataset.csv',
#                  o='/data/bioinf/projects/data/2020_deepFPlearn/modeltraining/ACoutside2/',
#                  t='smiles',
#                  k='topological',
#                  e=2, # 2000,
#                  s=2048,
#                  d=256,
#                  a=None,  # '/data/bioinf/projects/data/2020_deepFPlearn/modeltraining/ACoutside/ACmodel.hdf5',
#                  m=False,
#                  l=0.2,
#                  K=5,
#                  v=2)


# ------------------------------------------------------------------------------------- #
## The function defining what happens in the main training procedure 

def train(args: Namespace) -> None:

    logfile = args.o + 'training.log'
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG)

    # generate X and Y matrices
    dfInput = fp.prepareInputData(csvfilename=args.i, fp_size=args.s)

    encoder = dfpl.useOrTrainAutoencoder(data=dfInput, outpath=args.o, epochs=args.e,
                          encdim=args.d, verbosity=args.v, log=logfile)

    xcompressed = pd.DataFrame(encoder.predict(np.array(dfInput[dfInput['fp'].notnull()]['fp'].to_list())))
    # how can I add this to the dfInput?
    # I would like to provide dfInput to the following training procedures instead of x and y matrices
    ymatrix = dfInput[[col for col in dfInput.columns if col not in ['smiles', 'fp']]]

    # train FNNs with compressed features
    dfpl.trainNNmodels(modelfilepathprefix=args.o + "/FFNmodelACincl",
                      x=xcompressed, y=ymatrix,
                      split=args.l,
                      epochs=args.e, kfold=args.K, verbose=args.v)

    # train FNNs with uncompressed features

    ### train multi-label models
    # with comrpessed features
    dfpl.trainNNmodelsMulti(modelfilepathprefix=args.o + "/FNNmultiLabelmodelACincl",
                            x=xcompressed, y=ymatrix,
                            split=args.l, epochs=args.e,
                            verbose=args.v, kfold=args.K)

    # with uncompressed features
    dfpl.trainNNmodelsMulti(modelfilepathprefix=args.o + "/FNNmultiLabelmodelNoACincl",
                            x=xmatrix, y=ymatrix,
                            split=args.l, epochs=args.e,
                            verbose=args.v, kfold=args.K)


# ------------------------------------------------------------------------------------- #
## The function defining what happens in the main predict procedure 

def predict(args: Namespace) -> None:
    # generate X matrix
    (xpd, ymatrix) = dfpl.XandYfromInput(csvfilename=args.i, rtype=args.t, fptype=args.k,
                                         printfp=True, size=args.s, verbose=args.v, returnY=False)
    # predict values for provided data and model
    # ypredictions = dfpl.predictValues(modelfilepath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/2019-10-16_311681247_1000/model.Aromatase.h5", pdx=xpd)
    ypredictions = dfpl.predictValues(acmodelfilepath=args.ACmodel, modelfilepath=args.model, pdx=xpd)

    # write predictions to usr provided .csv file
    pd.DataFrame.to_csv(ypredictions, args.o)


# ===================================================================================== #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='deepFPlearn')
    subparsers = parser.add_subparsers(help="Sub programs of deepFPlearn")

    parser_train = subparsers.add_parser('train', help="Train new models with your data")
    parser_train.set_defaults(func=train)
    dfpl.parseInputTrain(parser_train)

    parser_predict = subparsers.add_parser('predict', help="Predict your data with existing models")
    parser_predict.set_defaults(func=predict)
    dfpl.parseInputPredict(parser_predict)

    args = parser.parse_args()
    print(f'[INFO] The following arguments are received or filled with default values:\n{args}')

    args.func(args)
