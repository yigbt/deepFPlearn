from argparse import Namespace
import pandas as pd
import importlib
import pathlib

from dfpl import dfpl
from dfpl import options

importlib.reload(dfpl)

project_directory = pathlib.Path(__file__).parent.parent.absolute()
test_train_args = options.TrainOptions(
    inputFile=f"{project_directory}/data/Sun_etal_dataset.csv",
    outputDir=f"{project_directory}/modeltraining",
    acFile="",
    type='smiles',
    fpType='topological',
    epochs=512,
    fpSize=2048,
    encFPSize=256,
    enableMultiLabel=False,
    testingFraction=0.2,
    kFolds=5,
    verbose=1
)


def train(opts: options.TrainOptions):
    """
    The function defining what happens in the main training procedure
    :param opts:
    """
    # generate X and Y matrices
    (xmatrix, ymatrix) = dfpl.XandYfromInput(
        csvfilename=opts.inputFile,
        rtype=opts.type,
        fptype=opts.fpType,
        printfp=True,
        size=opts.fpSize,
        verbose=opts.verbose
    )

    if opts.verbose > 0:
        print(f'[INFO] Shape of X matrix (input of AC/FNN): {xmatrix.shape}')
        print(f'[INFO] Shape of Y matrix (output of AC/FNN): {ymatrix.shape}')


    encoder = None
    if opts.acFile == "":
        # load trained model for autoencoder
        encoder = dfpl.trainfullac(X=xmatrix, y=ymatrix, epochs=args.e, encdim=args.d,
                                   useweights=args.a, verbose=args.v)
    else:
        # train an autoencoder on the full feature matrix
        encoder = dfpl.trainfullac(X=xmatrix, y=ymatrix, epochs=args.e, encdim=args.d,
                                   checkpointpath=args.o + "/ACmodel.hdf5",
                                   verbose=args.v)

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
    # dfpl.trainNNmodelsMulti(modelfilepathprefix=opts.o + "/FNNmultiLabelmodelNoACincl",
    #                         x=xmatrix, y=ymatrix,
    #                         split=opts.l, epochs=opts.e,
    #                         verbose=opts.v, kfold=opts.K)


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
    parser = options.createCommandlineParser()
    prog_args = parser.parse_args()
    print(f"[INFO] The following arguments are received or filled with default values:\n{prog_args}")

    if prog_args.method == "train":
        train(options.TrainOptions.fromCmdArgs(prog_args))
        exit(0)
    if prog_args.method == "predict":
        predict(prog_args)
        exit(0)
    # Fallthrough
    print("uhh, what happened here?")
