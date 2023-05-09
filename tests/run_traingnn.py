import unittest
import os
import shutil
import pathlib
import logging
import os
import sys
# Add the parent directory of the tests directory to the module search path
tests_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(tests_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, "./dfpl_chemprop")
from dfpl_chemprop import chemprop
from dfpl import __main__ as main
from dfpl import options


class TestTrainDMPNN(unittest.TestCase):

    def setUp(self):
        self.train_opts = options.GnnOptions(
            data_path="example/data/S_dataset-AR.csv",
            gnn_type="dmpnn",
            # save_dir="dmpnn-weight-AR/",
            # epochs=20,
            # num_folds=1,
            # metric="binary_cross_entropy",
            # extra_metrics="auc",
            # split_type="molecular_weight",
            # dataset_type="classification",
            # smiles_columns="smiles"
        )
        self.test_dir = "test_dmpnn"
        os.makedirs(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_train_dmpnn(self):
        main.traindmpnn(self.train_opts)
        # check that model weight file exists in save_dir
        self.assertTrue(os.path.isfile(os.path.join(self.train_opts.save_dir, "saved_model.pth")))
        # check that log file exists in save_dir
        self.assertTrue(os.path.isfile(os.path.join(self.train_opts.save_dir, "train.log")))


if __name__ == '__main__':
    unittest.main()
