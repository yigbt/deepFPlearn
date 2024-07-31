import logging
import os
import pickle

from sklearn.preprocessing import MinMaxScaler


def normalize_acc_values(df, column_name='AR', output_dir='.'):
    """
    Normalize ACC values in the dataframe and save the scaler.

    :param df: pandas DataFrame containing the data
    :param column_name: Name of the column to normalize
    :param output_dir: Directory to save the scaler
    :return: pandas DataFrame with normalized values
    """
    logging.info("Normalizing ACC values...")
    print("Normalizing ACC values...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    acc_values = df[column_name].values.reshape(-1, 1)
    scaled_acc_values = scaler.fit_transform(acc_values)
    df[column_name] = scaled_acc_values

    # Save the scaler for inverse transformation during prediction
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {scaler_path}")
    print(f"Scaler saved to {scaler_path}")

    return df, scaler_path

def inverse_transform_predictions(prediction, scaler_path):


    if os.path.exists(scaler_path):
        logging.info(f"Loading scaler from {scaler_path}")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logging.info("Applying inverse transformation to get pre-normalized values")
        return scaler.inverse_transform(prediction.reshape(-1, 1))
    else:
        logging.warning(f"Scaler file not found at {scaler_path}. Skipping normalization step.")
        return prediction
