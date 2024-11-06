import logging
import os

import numpy as np
import pandas as pd

from dfpl import options, settings
from dfpl import single_label_model as sl


def predict_values(df: pd.DataFrame, opts: options.Options) -> pd.DataFrame:
    """
    Predict a set of chemicals using a selected model.

    :param df: Input DataFrame containing the features (either compressed or uncompressed).
    :param opts: Model options including paths, feature types, and prediction preferences.
    :return: DataFrame with predictions.
    """

    # Determine the correct feature column and input size
    feature_column = "fpcompressed" if opts.compressFeatures else "fp"
    sub_df = df[df[feature_column].notnull()]

    if sub_df.empty:
        logging.warning(f"No valid features found in column '{feature_column}'")
        return pd.DataFrame()

    # Prepare the feature matrix for prediction
    x = np.array(
        sub_df[feature_column].to_list(),
        dtype=settings.nn_fp_compressed_numpy_type
        if opts.compressFeatures
        else settings.nn_fp_numpy_type,
        copy=settings.numpy_copy_values,
    )
    logging.info(
        f"{'Compressed' if opts.compressFeatures else 'Uncompressed'} FP matrix with shape {x.shape} and type {x.dtype}"
    )

    # Define the model architecture based on the feature size
    feature_input_size = x.shape[1]
    model = sl.define_single_label_model(input_size=feature_input_size, opts=opts)

    # Load the model weights
    weights_path = os.path.join(opts.fnnModelDir, "model_weights.hdf5")
    model.load_weights(weights_path)
    logging.info(f"Model weights loaded from {weights_path}")

    # Make predictions
    predictions = model.predict(x)

    # Add predictions to the DataFrame
    sub_df["predicted"] = predictions

    return sub_df
