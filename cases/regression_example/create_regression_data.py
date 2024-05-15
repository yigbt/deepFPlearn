import os
import pathlib
import pandas as pd
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename",
                        help="Path to CSV file containing classification train data which shall be randomly "
                             "transformed to regression data")
    args = parser.parse_args()
    return args


def main():
    """
    Reads the 'example/train_data.csv' file containing train data. Generates random values in (0, 2] for active
    compounds (class: 1) and 0 for inactive compounds. Values are replaced. The original column names are kept. The
    result is written to a new CSV file 'example/train_data_reg.csv' where '_reg' stands for regression.

    Returns:
        None
    """
    project_directory = pathlib.Path(".").parent.parent.absolute()

    args = get_args()

    df = pd.read_csv(os.path.join(project_directory, args.filename))

    df['AR'] = np.where(df['AR'] == 1, np.random.uniform(0, 2, df.shape[0]), 0)
    df['ER'] = np.where(df['ER'] == 1, np.random.uniform(0, 2, df.shape[0]), 0)
    df['GR'] = np.where(df['GR'] == 1, np.random.uniform(0, 2, df.shape[0]), 0)

    outfile = os.path.join(project_directory, f"{os.path.splitext(args.filename)[0]}_reg.csv")
    df.to_csv(path_or_buf=outfile, index=False)
    print(f"Data written to '{outfile}'")

    return


if __name__ == '__main__':
    main()
