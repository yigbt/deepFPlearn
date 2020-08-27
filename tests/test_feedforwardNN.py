import pathlib
import os.path
import numpy as np

import dfpl.fingerprint as fp
import dfpl.feedforwardNN as fNN
import dfpl.options as opts


def test_prepare_nn_training_data():
    project_directory = pathlib.Path(__file__).parent.parent.absolute()
    df = fp.importDataFile(os.path.join(project_directory, "data", "Sun_etal_dataset.csv"))

    targets = ["AR", "ER", "GR", "Aromatase", "TR", "PPARg"]
    fractions = [0.5, 1.0, 2.0, 3.0]
    for f in fractions:
        o = opts.TrainOptions(
            compressFeatures=False,
            sampleFractionOnes=f
        )
        for t in targets:
            x, y = fNN.prepare_nn_training_data(df, t, o)
            unique, counts = np.unique(y, return_counts=True)
            assert abs(counts[1]/counts[0] - f) < 0.01
            print(f"Wanted \"{t}\" fraction: {f}, got sampling: {dict(zip(unique, counts))}, Result fraction: {counts[1]/counts[0]}")

