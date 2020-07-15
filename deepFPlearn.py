import argparse
from argparse import Namespace
import pandas as pd

import dfplmodule as dfpl
import importlib
import pathlib

importlib.reload(dfpl)

project_directory = pathlib.Path(__file__).parent.absolute()
test_train_args = Namespace(
    i=f"{project_directory}/testdata/TrainingDataset.csv",
    o=f"{project_directory}/modeltraining",
    t='smiles',
    k='topological',
    e=512,
    s=2048,
    d=256,
    a=None,
    m=False,
    l=0.2,
    K=5,
    v=0
)


def train(args):
    """
    The function defining what happens in the main training procedure
    :param args:
    """
    # generate X and Y matrices
    (xmatrix, ymatrix) = dfpl.XandYfromInput(csvfilename=args.i, rtype=args.t, fptype=args.k,
                                             printfp=True, size=args.s, verbose=args.v)

    # if args.v > 0:
    #     print(f'[INFO] Shape of X matrix (input of AC/FNN): {xmatrix.shape}')
    #     print(f'[INFO] Shape of Y matrix (output of AC/FNN): {ymatrix.shape}')
    #
    #
    # encoder = None
    # if args.a:
    #     # load trained model for autoencoder
    #     encoder = dfpl.trainfullac(X=xmatrix, y=ymatrix, epochs=args.e, encdim=args.d,
    #                                useweights=args.a, verbose=args.v)
    # else:
    #     # train an autoencoder on the full feature matrix
    #     encoder = dfpl.trainfullac(X=xmatrix, y=ymatrix, epochs=args.e, encdim=args.d,
    #                                checkpointpath=args.o + "/ACmodel.hdf5",
    #                                verbose=args.v)
    #
    # xcompressed = pd.DataFrame(data=encoder.predict(xmatrix))

    # # train FNNs with compressed features
    # dfpl.trainNNmodels(modelfilepathprefix=args.o + "/FFNmodelACincl",
    #                    x=xcompressed, y=ymatrix,
    #                    split=args.l,
    #                    epochs=args.e, kfold=args.K, verbose=args.v)
    #
    # # train FNNs with uncompressed features
    # dfpl.trainNNmodels(modelfilepathprefix=args.o + "/FFNmodelNoACincl",
    #                    x=xmatrix, y=ymatrix,
    #                    split=args.l,
    #                    epochs=args.e, kfold=args.K, verbose=args.v)

    ### train multi-label models
    # with comrpessed features
    # dfpl.trainNNmodelsMulti(modelfilepathprefix=args.o + "/FNNmultiLabelmodelACincl",
    #                         x=xcompressed, y=ymatrix,
    #                         split=args.l, epochs=args.e,
    #                         verbose=args.v, kfold=args.K)

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

    parser_train = subparsers.add_parser("train", help="Train new models with your data")
    parser_train.set_defaults(func=train)
    dfpl.parseInputTrain(parser_train)

    parser_predict = subparsers.add_parser("predict", help="Predict your data with existing models")
    parser_predict.set_defaults(func=predict)
    dfpl.parseInputPredict(parser_predict)

    prog_args = parser.parse_args()
    print(f"[INFO] The following arguments are received or filled with default values:\n{prog_args}")

    prog_args.func(prog_args)
#    print(args)
