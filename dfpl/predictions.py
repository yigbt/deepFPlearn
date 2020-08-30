import pandas as pd
import numpy as np

import options
import feedforwardNN as fNN
import settings


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
            dtype=settings.ac_fp_compressed_numpy_type,
            copy=settings.numpy_copy_values
        )
    else:
        x = np.array(
            df[df['fp'].notnull()]['fp'].to_list(),
            dtype=settings.ac_fp_numpy_type,
            copy=settings.numpy_copy_values
        )

    model = fNN.define_nn_model(input_size=x.shape[1])
    predictions_random = pd.DataFrame(model.predict(x), columns=['random'])
    model.load_weights(opts.model)
    predictions = pd.DataFrame(model.predict(x), columns=['trained'])
    return df.join(predictions_random.join(predictions))
