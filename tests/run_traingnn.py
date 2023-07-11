import logging
import os
import pathlib
import sys

tests_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(tests_dir)
sys.path.insert(0, parent_dir)
import chemprop as cp
from chemprop import args

import dfpl.utils as utils

project_directory = pathlib.Path(__file__).parent.absolute()
test_train_args = args.TrainArgs(
    inputFile=utils.makePathAbsolute(f"{project_directory}/data/S_dataset.csv"),
)


def test_traindmpnn(opts: args.TrainArgs) -> None:
    print("Running traindmpnn test...")

    print("Training DMPNN...")
    mean_score, std_score = cp.train.cross_validate(
        args=opts, train_func=cp.train.run_training
    )

    print(f"Results: {mean_score:.5f} +/- {std_score:.5f}")
    print("traindmpnn test complete.")


if __name__ == "__main__":
    test_traindmpnn(test_train_args)
