import logging
from time import time
import numpy as np
import pandas as pd
# import tensorflow.keras.models
import keras.models
# import shap
from sklearn.inspection import permutation_importance

from dfpl import options


def get_feature_importance(model: keras.models.Model,
                           x_train: np.ndarray,
                           y_train: np.ndarray
                           ) -> tuple:
    """

    :param y_train:
    :param x_train:
    :param model:
    :return:
    """

    logging.info(
        f"Doing feature importance calculation"
    )
    start = time()
    perm_importance = permutation_importance(estimator=model,
                                             X=x_train,
                                             y=y_train,
                                             scoring="neg_mean_squared_error")
    explain_time = str(round((time() - start) / 60, ndigits=2))
    logging.info(
        f"Computation time for feature importance: {explain_time} min"
    )

    return perm_importance.importances_mean


def get_important_feature_indices(feature_importance: np.ndarray,
                                  threshold: float,
                                  top_x: 512) -> np.ndarray:
    feature_importance_top_x = np.sort(feature_importance)[::-1][:top_x]

    return np.where(np.isin(feature_importance, feature_importance_top_x))[0]


# def get_shapley_values(model: keras.models.Model,
#                        input_data: np.array,
#                        background_data: np.array) -> np.array:
#
#     # create a SHAP explainer object
#     explainer = shap.Explainer(model=model)
#
#     # compute shap values only for important features
#     shap_values = explainer.shap_values(input_data)
#
#     return shap_values


# this function is not used so far!
def get_explainable(df: pd.DataFrame,
                    opts: options.Options,
                    feature_importance_threshold: 0.1,
                    background_data_fraction: 0.8) -> pd.DataFrame:
    model = keras.models.load_model(opts.fnnModelDir)

    feature_importance = []  # get_feature_importance(model=model, x_train=, y_train=)

    # Get indices of the important features
    important_features_indices = np.where(feature_importance=feature_importance_threshold)

    # shapley_values = get_shapley_values(indices=important_features_indices,
    #                                     bg_data_fraction=background_data_fraction,
    #                                     model=model,
    #                                     df=df)

    return pd.DataFrame(
        {
            'Index': important_features_indices,
            'FeatureImportance': feature_importance[important_features_indices],
            # 'SHAPleyValues': shapley_values
        }
    )
