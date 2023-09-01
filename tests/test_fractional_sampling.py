import os.path
import pathlib

import numpy as np

from dfpl import fingerprint as fp
from dfpl import options as opts
from dfpl import single_label_model as fNN


def test_fractional_sampling():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the data directory
    data_directory = os.path.join(script_directory, "data")
    df = fp.importDataFile(os.path.join(data_directory, "S_dataset.csv"))

    # test_directory = pathlib.Path(__file__).parent.absolute()
    # df = fp.importDataFile(os.path.join(test_directory, "data", "S_dataset.csv"))

    targets = ["AR", "ER", "GR"]
    fractions = [0.5, 1.0, 2.0, 3.0]
    for f in fractions:
        o = opts.Options(compressFeatures=False, sampleFractionOnes=f, sampleDown=True)
        for t in targets:
            x, y = fNN.prepare_nn_training_data(df, t, o)
            if x is not None:
                unique, counts = np.unique(y, return_counts=True)
                assert abs(counts[1] / counts[0] - f) < 0.01
                print(
                    f'Wanted "{t}" fraction: {f}, got sampling: {dict(zip(unique, counts))}, '
                    f"Result fraction: {counts[1] / counts[0]}"
                )
