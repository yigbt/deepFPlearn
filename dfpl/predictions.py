import pandas as pd
import numpy as np
import logging

from dfpl import options
from dfpl import feedforwardNN as fNN
from dfpl import settings


def predict_values(df: pd.DataFrame,
                   opts: options.PredictOptions,
                   use_compressed: bool) -> pd.DataFrame:
    """
    Predict a set of chemicals using a selected model.

    :param df:
    :param opts:
    :param use_compressed:
    :return:
    """

    if use_compressed:
        x = np.array(
            df[df['fpcompressed'].notnull()]['fpcompressed'].to_list(),
            dtype=settings.nn_fp_compressed_numpy_type,
            copy=settings.numpy_copy_values
        )
        logging.info(f"Compressed FP matrix with shape {x.shape} and type {x.dtype}")
    else:
        x = np.array(
            df[df['fp'].notnull()]['fp'].to_list(),
            dtype=settings.nn_fp_numpy_type,
            copy=settings.numpy_copy_values
        )
        logging.info(f"Uncompressed FP matrix with shape {x.shape} and type {x.dtype}")

    model = fNN.define_nn_model(input_size=x.shape[1])
    logging.info("Predicting with random model weights")
    predictions_random = pd.DataFrame(model.predict(x), columns=['random'])
    model.load_weights(opts.model)
    logging.info("Predicting with trained model weights")
    predictions = pd.DataFrame(model.predict(x), columns=['trained'])
    return df.join(predictions_random.join(predictions))
