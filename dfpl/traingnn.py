from __future__ import annotations

import argparse
import logging
from argparse import Namespace
from dataclasses import dataclass
from typing import List

import chemprop
from chemprop.args import TrainArgs

from dfpl.utils import createLogger


@dataclass
class GnnOptions(TrainArgs):
    """
    Dataclass to hold all options used for training the graph models
    """

    total_epochs: int = 30
    save: bool = True
    # configFile: str = "./example/traingnn.json"
    data_path: str = "./example/data/tox21.csv"
    use_compound_names: bool = False
    save_dir: str = ""
    no_cache: bool = False
    features_scaling: bool = True
    use_input_features: str = ""
    cuda: bool = False
    num_lrs: int = 2
    minimize_score: bool = False
    num_tasks: int = 12
    preds_path: str = "./tox21dmpnn.csv"
    test_path: str = ""
    save_preds: bool = True
    calibration_method: str = "none"
    uncertainty_method: str = "none"
    calibration_path: str = ""
    evaluation_methods: str = "none"
    evaluation_scores_path: str = ""
    wabTracking: bool = False
    split_sizes: List[float] = None

    # save_smiles_splits: bool = False


def parseTrainGnn(parser_train_gnn: argparse.ArgumentParser) -> None:
    train_gnn_general_args = parser_train_gnn.add_argument_group("General Configuration")
    train_gnn_data_args = parser_train_gnn.add_argument_group("Data Configuration")
    train_gnn_files_args = parser_train_gnn.add_argument_group("Files")
    train_gnn_model_args = parser_train_gnn.add_argument_group("Model arguments")
    train_gnn_training_args = parser_train_gnn.add_argument_group("Training Configuration")
    train_gnn_uncertainty_args = parser_train_gnn.add_argument_group("Uncertainty Configuration")
    train_gnn_uncertainty_args.add_argument(
        "--uncertainty_method",
        type=str,
        choices=[
            "mve",
            "ensemble",
            "evidential_epistemic",
            "evidential_aleatoric",
            "evidential_total",
            "classification",
            "dropout",
            "dirichlet",
        ],
        help="Method to use for uncertainty estimation",
        default="none",
    )
    # Uncertainty arguments
    train_gnn_uncertainty_args.add_argument(
        "--calibration_method",
        type=str,
        choices=[
            "zscaling",
            "tscaling",
            "zelikman_interval",
            "mve_weighting",
            "platt",
            "isotonic",
        ],
        help="Method to use for calibration",
        default="none",
    )
    train_gnn_uncertainty_args.add_argument(
        "--calibration_path",
        type=str,
        metavar="FILE",
        help="Path to file with calibration data",
    )

    # General arguments
    train_gnn_general_args.add_argument(
        "--split_key_molecule",
        type=int,
        help="The index of the key molecule used for splitting",
    )
    train_gnn_general_args.add_argument("--pytorch_seed", type=int, help="Seed for pytorch")
    train_gnn_general_args.add_argument(
        "--cache_cutoff",
        type=float,
        help="Maximum number of molecules in dataset to allow caching.",
    )
    train_gnn_general_args.add_argument(
        "--save_preds", help="Saves test split predictions during training", type=bool
    )
    train_gnn_general_args.add_argument("--wabTracking", action="store_true", default=False)
    train_gnn_general_args.add_argument(
        "--cuda", action="store_true", default=False, help="Turn on cuda"
    )
    train_gnn_general_args.add_argument(
        "--save_smiles_splits",
        action="store_true",
        default=False,
        help="Save smiles for each train/val/test splits",
    )
    train_gnn_general_args.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether to skip training and only test the model",
    )
    train_gnn_general_args.add_argument(
        "--gpu",
        type=int,
        # choices=list(range(torch.cuda.device_count())),
        help="Which GPU to use",
    )
    train_gnn_general_args.add_argument("--save", type=bool)
    train_gnn_general_args.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Skip non-essential print statements",
    )
    train_gnn_general_args.add_argument(
        "--log_frequency",
        type=int,
        metavar="INT",
        default=10,
        help="The number of batches between each log",
    )
    train_gnn_general_args.add_argument(
        "--no_cache_mol",
        action="store_true",
        default=False,
        help="If raised, Turn off caching rdkit mols",
    )

    # FILES ARGUMENTS
    train_gnn_files_args.add_argument(
        "--save_dir",
        type=str,
        metavar="DIR",
        default="./ckpt/",
        help="Directory where model checkpoints will be saved",
    )
    train_gnn_files_args.add_argument(
        "--checkpoint_dir",
        type=str,
        metavar="DIR",
        default=None,
        help="Directory from which to load model checkpoints"
        "(walks directory and ensembles all models that are found)",
    )
    train_gnn_files_args.add_argument(
        "--checkpoint_path",
        type=str,
        metavar="FILE",
        default=None,
        help="Path to model checkpoint (.pt file)",
    )
    train_gnn_files_args.add_argument(
        "--checkpoint_paths",
        type=str,
        metavar="FILE",
        nargs="*",
        default=None,
        help="Path to model checkpoint (.pt file)",
    )
    train_gnn_files_args.add_argument(
        "--separate_val_path",
        type=str,
        metavar="FILE",
        help="Path to separate val set, optional",
    )
    train_gnn_files_args.add_argument(
        "--separate_val_features_path",
        type=str,
        metavar="FILE",
        nargs="*",
        help="Path to file with features for separate val set",
    )
    train_gnn_files_args.add_argument(
        "--separate_test_path",
        type=str,
        metavar="FILE",
        help="Path to separate test set, optional",
    )
    train_gnn_files_args.add_argument(
        "--separate_test_features_path",
        type=str,
        metavar="FILE",
        nargs="*",
        help="Path to file with features for separate test set",
    )
    train_gnn_files_args.add_argument(
        "--folds_file",
        type=str,
        metavar="FILE",
        default=None,
        help="Optional file of fold labels",
    )
    train_gnn_files_args.add_argument(
        "--val_fold_index",
        type=int,
        metavar="INT",
        default=None,
        help="Which fold to use as val for cross val",
    )
    train_gnn_files_args.add_argument(
        "--test_fold_index",
        type=int,
        metavar="INT",
        default=None,
        help="Which fold to use as test for cross val",
    )
    train_gnn_files_args.add_argument(
        "--crossval_index_dir",
        type=str,
        metavar="DIR",
        help="Directory in which to find cross validation index files",
    )
    train_gnn_files_args.add_argument(
        "--crossval_index_file",
        type=str,
        metavar="FILE",
        help="Indices of files to use as train/val/test"
        "Overrides --num_folds and --seed.",
    )
    train_gnn_files_args.add_argument(
        "--data_weights_path",
        type=str,
        metavar="FILE",
        help="Path where the data weight are saved",
    )
    train_gnn_files_args.add_argument(
        "--features_path",
        type=str,
        metavar="FILE",
        nargs="*",
        help="Path to features to use in FNN (instead of features_generator)",
    )

    train_gnn_files_args.add_argument(
        "--separate_val_phase_features_path", type=str, metavar="FILE"
    )
    train_gnn_files_args.add_argument(
        "--separate_test_phase_features_path", type=str, metavar="FILE"
    )

    train_gnn_files_args.add_argument(
        "--separate_val_atom_descriptors_path", type=str, metavar="FILE"
    )
    train_gnn_files_args.add_argument(
        "--separate_test_atom_descriptors_path", type=str, metavar="FILE"
    )
    # Data related arguments
    train_gnn_data_args.add_argument(
        "--data_path",
        type=str,
        metavar="FILE",
        help="Path to data CSV file",
        default="",
    )
    train_gnn_data_args.add_argument(
        "--use_compound_names",
        action="store_true",
        default=False,
        help="Use when test data file contains compound names in addition to SMILES strings",
    )
    train_gnn_data_args.add_argument(
        "--max_data_size",
        type=int,
        metavar="INT",
        help="Maximum number of data points to load",
    )

    train_gnn_data_args.add_argument(
        "--features_only",
        action="store_true",
        default=False,
        help="Use only the additional features in an FFN, no graph network",
    )
    train_gnn_data_args.add_argument(
        "--dataset_type",
        type=str,
        choices=["classification", "regression", "multiclass"],
        help="Type of dataset, e.g. classification or regression."
        "This determines the loss function used during training.",
        default="regression",
    )  # classification
    train_gnn_data_args.add_argument(
        "--multiclass_num_classes",
        type=int,
        metavar="INT",
        default=3,
        help="Number of classes in multiclass classification",
    )
    train_gnn_data_args.add_argument(
        "--split_type",
        type=str,
        default="random",
        choices=[
            "random",
            "scaffold_balanced",
            "predetermined",
            "crossval",
            "index_predetermined",
        ],
        help="Method of splitting the data into train/val/test",
    )
    train_gnn_data_args.add_argument(
        "--split_sizes",
        type=float,
        metavar="FLOAT",
        nargs=3,
        default=[0.8, 0.2, 0.0],
        help="Split proportions for train/validation/test sets",
    )

    train_gnn_data_args.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed to use when splitting data into train/val/test sets."
        "When `num_folds` > 1, the first fold uses this seed and all"
        "subsequent folds add 1 to the seed.",
    )
    train_gnn_data_args.add_argument(
        "--smiles_columns",
        type=str,
        metavar="STRING",
        help="Name of the smiles columns",
    )

    train_gnn_data_args.add_argument(
        "--target_columns",
        type=str,
        nargs="*",
        metavar="STRING",
        help="Name of the target columns",
    )

    train_gnn_data_args.add_argument(
        "--ignore_columns",
        type=str,
        nargs="*",
        metavar="STRING",
        help="Names of the columns to ignore",
    )
    train_gnn_data_args.add_argument(
        "--num_tasks", type=int, metavar="INT", help="Number of tasks"
    )
    train_gnn_data_args.add_argument(
        "--no_features_scaling",
        action="store_true",
        default=False,
        help="Turn off scaling of features",
    )
    train_gnn_data_args.add_argument(
        "--features_scaling",
        action="store_true",
        default=False,
        help="Turn on scaling of features",
    )
    train_gnn_data_args.add_argument(
        "--use_input_features",
        type=str,
        metavar="STRING",
        help="Turn on scaling of features",
    )

    # Model arguments
    train_gnn_model_args.add_argument(
        "--ensemble_size",
        type=int,
        metavar="INT",
        default=1,
        help="Number of models in ensemble",
    )
    train_gnn_model_args.add_argument(
        "--hidden_size",
        type=int,
        metavar="INT",
        default=300,
        help="Dimensionality of hidden layers in MPN",
    )
    train_gnn_model_args.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Whether to add bias to linear layers",
    )
    train_gnn_model_args.add_argument(
        "--depth",
        type=int,
        metavar="INT",
        default=3,
        help="Number of message passing steps",
    )
    train_gnn_model_args.add_argument(
        "--dropout",
        type=float,
        metavar="FLOAT",
        default=0.0,
        help="Dropout probability",
    )
    train_gnn_model_args.add_argument(
        "--activation",
        type=str,
        default="ReLU",
        choices=["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"],
        help="Activation function",
    )
    train_gnn_model_args.add_argument(
        "--undirected",
        action="store_true",
        default=False,
        help="Undirected edges (always sum the two relevant bond vectors)",
    )
    train_gnn_model_args.add_argument(
        "--ffn_hidden_size",
        type=int,
        metavar="INT",
        default=2,
        help="Hidden dim for higher-capacity FFN (defaults to hidden_size)",
    )
    train_gnn_model_args.add_argument(
        "--ffn_num_layers",
        type=int,
        metavar="INT",
        default=2,
        help="Number of layers in FFN after MPN encoding",
    )
    train_gnn_model_args.add_argument(
        "--atom_messages",
        action="store_true",
        default=False,
        help="Use messages on atoms instead of messages on bonds",
    )

    train_gnn_model_args.add_argument(
        "--num_lrs",
        type=int,
        metavar="INT",
        default=2,
        help="Number of layers in FFN after MPN encoding",
    )
    train_gnn_model_args.add_argument(
        "--checkpoint_frzn", type=str, metavar="STRING", help="Freeze the loaded model"
    )
    # Model arguments
    # model_args.add_argument("--mpn_shared", type=bool, metavar="BOOL")
    train_gnn_model_args.add_argument(
        "--show_individual_scores",
        action="store_true",
        default=True,
        help="Show all scores for individual targets, not just average, at the end",
    )
    train_gnn_model_args.add_argument(
        "--aggregation",
        choices=["mean", "sum", "norm"],
        help="Aggregation scheme for atomic vectors into molecular vectors",
    )
    train_gnn_model_args.add_argument(
        "--aggregation_norm",
        type=int,
        help="For norm aggregation, number by which to divide summed up atomic features",
    )
    # model_args.add_argument("--explicit_h", type=bool, metavar="BOOL",help="A explicit hydrogen")
    train_gnn_model_args.add_argument(
        "--adding_h", type=bool, metavar="BOOL", help="Adding hydrogen"
    )
    # Training arguments
    train_gnn_model_args.add_argument(
        "--class_balance",
        type=bool,
        metavar="BOOL",
        help="Balances the classes across batches",
    )
    train_gnn_model_args.add_argument(
        "--evidential_regularization",
        type=float,
        metavar="FLOAT",
        help="Regularization parameter for evidential loss",
    )
    train_gnn_model_args.add_argument(
        "--overwrite_default_atom_features",
        type=bool,
        metavar="BOOL",
        help="Overwrites default atom features instead of concatenating",
    )
    train_gnn_model_args.add_argument("--no_atom_descriptor_scaling", type=bool, metavar="BOOL")
    train_gnn_model_args.add_argument(
        "--overwrite_default_bond_features",
        type=bool,
        metavar="BOOL",
        help="Overwrites default bond features instead of concatenating",
    )
    train_gnn_model_args.add_argument(
        "--frzn_ffn_layers",
        type=int,
        metavar="INT",
        help="Number of layers in FFN to freeze",
    )
    # model_args.add_argument("--freeze_first_only", type=bool, metavar="BOOL")
    # Training arguments
    train_gnn_training_args.add_argument(
        "--epochs", type=int, metavar="INT", default=30, help="Number of epochs to run"
    )
    train_gnn_training_args.add_argument(
        "--total_epochs",
        type=int,
        metavar="INT",
        default=30,
        help="Number of total epochs to run",
    )
    train_gnn_training_args.add_argument(
        "--batch_size", type=int, metavar="INT", default=50, help="Batch size"
    )
    train_gnn_training_args.add_argument(
        "--warmup_epochs",
        type=int,
        metavar="INT",
        default=2,
        help="Number of epochs during which learning rate increases linearly from"
        "init_lr to max_lr. Afterwards, learning rate decreases exponentially"
        "from max_lr to final_lr.",
    )
    train_gnn_training_args.add_argument(
        "--init_lr",
        type=float,
        metavar="FLOAT",
        default=1e-4,
        help="Initial learning rate",
    )
    train_gnn_training_args.add_argument(
        "--max_lr",
        type=float,
        metavar="FLOAT",
        default=1e-3,
        help="Maximum learning rate",
    )
    train_gnn_training_args.add_argument(
        "--final_lr",
        type=float,
        metavar="FLOAT",
        default=1e-4,
        help="Final learning rate",
    )
    train_gnn_training_args.add_argument(
        "--extra_metrics",
        type=str,
        metavar="STRING",
        nargs="*",
        help="Extra metrics to use",
    )
    train_gnn_training_args.add_argument(
        "--loss_function",
        type=str,
        choices=[
            "mse",
            "bounded_mse",
            "binary_cross_entropy",
            "cross_entropy",
            "mcc",
            "sid",
            "wasserstein",
            "mve",
            "evidential",
            "dirichlet",
        ],
    )
    train_gnn_training_args.add_argument(
        "--grad_clip", type=float, metavar="FLOAT", help="Gradient clipping value"
    )
    train_gnn_training_args.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=[
            "auc",
            "prc-auc",
            "rmse",
            "mae",
            "mse",
            "r2",
            "accuracy",
            "cross_entropy",
        ],
        help="Metric to use during evaluation."
        "Note: Does NOT affect loss function used during training"
        "(loss is determined by the `dataset_type` argument)."
        'Note: Defaults to "auc" for classification and "rmse" for regression.',
    )
    train_gnn_training_args.add_argument(
        "--num_folds",
        type=int,
        metavar="INT",
        default=1,
        help="Number of folds when performing cross validation",
    )


def traindmpnn(args: Namespace) -> None:
    """
    Train a D-MPNN model using the given options.
    """
    createLogger("traingnn.log")
    opts = GnnOptions(**vars(args))
    logging.info("Training DMPNN...")
    mean_score, std_score = chemprop.train.cross_validate(
        args=opts, train_func=chemprop.train.run_training
    )
    logging.info(f"Results: {mean_score:.5f} +/- {std_score:.5f}")
