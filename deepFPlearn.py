import argparse
from argparse import Namespace
import pandas as pd

import dfplmodule as dfpl
import importlib

importlib.reload(dfpl)

# args = Namespace(i='/data/bioinf/projects/data/2020_deepFPlearn/dataSources/Sun_et_al/Sun_etal_dataset.csv',
#                        o='/data/bioinf/projects/data/2020_deepFPlearn/modeltraining/ACoutside2/',
#                        t='smiles',
#                        k='topological',
#                        e=11,  # 2000,
#                        s=2048,
#                        d=256,
#                        a=None,  # '/data/bioinf/projects/data/2020_deepFPlearn/modeltraining/ACoutside/ACmodel.hdf5',
#                        m=False,
#                        l=0.2,
#                        K=5,
#                        v=2)


# ------------------------------------------------------------------------------------- #
## The function defining what happens in the main training procedure 

def train(args: Namespace) -> None:

    # generate X and Y matrices
    (xmatrix, ymatrix) = dfpl.XandYfromInput(csvfilename=args.i, rtype=args.t, fptype=args.k,
                                             printfp=True, size=args.s, verbose=args.v)

    if args.v > 0:
        print(f'[INFO] Shape of X matrix (input of AC/FNN): {xmatrix.shape}')
        print(f'[INFO] Shape of Y matrix (output of AC/FNN): {ymatrix.shape}')

    encoder = None
    if args.a:
        # load trained model for autoencoder
        # this is not working yet.. I cannot load the AC weights into the encoder model (of course not!)
        # But i don't have a solution for that at the moment
        encoder = dfpl.trainfullac(X=xmatrix, y=ymatrix, epochs=args.e, encdim=args.d,
                                   useweights=args.a, verbose=args.v)
    else:
        # train an autoencoder on the full feature matrix
        weightfileAC = args.o + "/ACmodel.autoencoder.hdf5"
        encoder = dfpl.trainfullac(X=xmatrix, y=ymatrix, epochs=args.e, encdim=args.d,
                                   checkpointpath=weightfileAC,
                                   verbose=args.v)
        weightfile = args.o + "/ACmodel.encoder.hdf5"
        encoder.save_weights(weightfile)

        if args.ACtrainOnly:
            print(f'[INFO] Model weights of trained autencoder are stored in: {weightfile}')
            exit(1)

    xcompressed = pd.DataFrame(data=encoder.predict(xmatrix))

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
