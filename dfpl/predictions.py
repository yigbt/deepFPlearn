import logging

import numpy as np
import pandas as pd
import tensorflow.keras.models

from dfpl import options, settings


def predict_values(df: pd.DataFrame, opts: options.Options) -> pd.DataFrame:
    """
    Predict a set of chemicals using a selected model.

    :param df:
    :param opts:
    :return:
    """
    model = tensorflow.keras.models.load_model(opts.fnnModelDir, compile=False)
    if opts.compressFeatures:
        sub_df = df[df["fpcompressed"].notnull()]
        x = np.array(
            sub_df["fpcompressed"].to_list(),
            dtype=settings.nn_fp_compressed_numpy_type,
            copy=settings.numpy_copy_values,
        )
        logging.info(f"Compressed FP matrix with shape {x.shape} and type {x.dtype}")
        sub_df["predicted"] = pd.DataFrame(model.predict(x), columns=["predicted"])
        return sub_df
    else:
        sub_df = df[df["fp"].notnull()]
        x = np.array(
            sub_df["fp"].to_list(),
            dtype=settings.nn_fp_numpy_type,
            copy=settings.numpy_copy_values,
        )
        logging.info(f"Uncompressed FP matrix with shape {x.shape} and type {x.dtype}")
        sub_df["predicted"] = pd.DataFrame(model.predict(x), columns=["predicted"])
        return sub_df
