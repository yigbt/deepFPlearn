import pathlib
import os
from matplotlib import pyplot as plt
import json
import pandas as pd


def makePathAbsolute(p: str) -> str:
    path = pathlib.Path(p)
    if path.is_absolute():
        return p
    else:
        return str(path.absolute())


def createDirectory(directory: str):
    path = makePathAbsolute(directory)
    if not os.path.exists(path):
        os.makedirs(path)


def makePlots(save_path: str, training_auc: list, training_loss: list, validation_auc: list, validation_loss: list):

    all_data = [training_auc, training_loss, validation_auc, validation_loss]
    zipped = list(zip(training_loss, validation_loss, training_auc, validation_auc))

    metricsdf = pd.DataFrame(zipped, columns=['LOSS', 'VAL_LOSS', 'AUC', 'VAL_AUC'])
    metricsdf.to_csv(f"{save_path}/metrics.csv")

    metricsdf.plot(title='Model performance')
    plt.savefig(f"{save_path}/plot.png", format='png')

    lossesdf = pd.DataFrame(list(zip(training_loss,validation_loss)), columns=['LOSS', 'VAL_LOSS'])
    lossesdf.plot(title='Loss')
    plt.savefig(f"{save_path}/loss.png", format='png')

    aucdf = pd.DataFrame(list(zip(training_auc,validation_auc)), columns=['AUC', 'VAL_AUC'])
    aucdf.plot(title='AUC')
    plt.savefig(f"{save_path}/auc.png", format='png')


def createArgsFromJson(in_json: str, ignore_elements: list, return_json_object: bool):
    arguments = []
    with open(in_json, 'r') as f:
        data = json.load(f)
    for i, j in data.items():
        if str(i) not in ignore_elements:
            i = "--" + str(i)
            j = str(j)
            arguments.append(i)
            arguments.append(j)
    if return_json_object:
        return arguments, data
    return arguments