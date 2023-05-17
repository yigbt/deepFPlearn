import unittest
import os
import shutil
import pathlib
import logging
import os
import sys
import json

# Add the parent directory of the tests directory to the module search path
tests_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(tests_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, "./dfpl_chemprop")
from dfpl_chemprop import chemprop
from dfpl_chemprop.chemprop import args
from dfpl import __main__ as main

from dfpl import options


class TestTrainDMPNN(unittest.TestCase):
    def setUp(self):
        self.opts_file = "./example/traingnn.json"
        with open(self.opts_file, "r") as f:
            opts_dict = json.load(f)
        ignore_elements = ["py/object", "gnn_type"]
        filtered_opts_dict = {
            k: v for k, v in opts_dict.items() if k not in ignore_elements
        }
        print(filtered_opts_dict)
        self.train_opts = args.TrainArgs().parse_args(filtered_opts_dict)
        self.save_dir = "test_dmpnn"
        os.makedirs(self.save_dir)

    def tearDown(self):
        shutil.rmtree(self.save_dir)

    def test_train_dmpnn(self):
        main.traindmpnn(self.train_opts)
        # check that model weight file exists in save_dir
        self.assertTrue(
            os.path.isfile(os.path.join(self.train_opts.save_dir, "saved_model.pth"))
        )
        # check that log file exists in save_dir
        self.assertTrue(
            os.path.isfile(os.path.join(self.train_opts.save_dir, "train.log"))
        )


if __name__ == "__main__":
    unittest.main()
