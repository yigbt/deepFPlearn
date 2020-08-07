import pathlib
import logging

import dfpl.options as opt
import dfpl.fingerprint as fp
import dfpl.autoencoder as ac

project_directory = pathlib.Path(__file__).parent.parent.absolute()
test_train_args = opt.TrainOptions(
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


p