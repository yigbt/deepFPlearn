from __future__ import annotations

import argparse
from argparse import Namespace
from typing import Optional, Literal, List

import chemprop
from chemprop.args import PredictArgs

from dfpl.utils import createLogger


class PredictGnnOptions(PredictArgs):
    """
    Dataclass to hold all options used for training the graph models
    """

    # configFile: str = "./example/predictgnn.json"
    calibration_atom_descriptors_path: str = None
    calibration_features_path: str = None
    calibration_interval_percentile: float = 95
    calibration_method: Optional[
        Literal[
            "zscaling",
            "tscaling",
            "zelikman_interval",
            "mve_weighting",
            "platt",
            "isotonic",
        ]
    ] = None
    calibration_path: str = None
    calibration_phase_features_path: str = None
    drop_extra_columns: bool = False
    dropout_sampling_size: int = 10
    evaluation_methods: List[str] = None
    evaluation_scores_path: str = None
    # no_features_scaling: bool = True
    individual_ensemble_predictions: bool = False
    preds_path: str = None
    regression_calibrator_metric: Optional[Literal["stdev", "interval"]] = None
    test_path: str = None
    uncertainty_dropout_p: float = 0.1
    uncertainty_method: Optional[
        Literal[
            "mve",
            "ensemble",
            "evidential_epistemic",
            "evidential_aleatoric",
            "evidential_total",
            "classification",
            "dropout",
        ]
    ] = None


def parsePredictGnn(parser_predict_gnn: argparse.ArgumentParser) -> None:
    predict_gnn_general_args = parser_predict_gnn.add_argument_group("General Configuration")
    predict_gnn_files_args = parser_predict_gnn.add_argument_group("Files")
    predict_gnn_uncertainty_args = parser_predict_gnn.add_argument_group("Uncertainty Configuration")

    predict_gnn_general_args.add_argument(
        "--checkpoint_path",
        type=str,
        metavar="FILE",
        help="Path to model checkpoint (.pt file)",
    )
    predict_gnn_files_args.add_argument(
        "--test_path",
        type=str,
        help="Path to CSV file for which predictions will be made.",
    )
    predict_gnn_files_args.add_argument(
        "--preds_path",
        type=str,
        help="Predictions output file. CSV or PICKLE file where predictions will be saved.",
    )
    predict_gnn_files_args.add_argument(
        "--calibration_path",
        type=str,
        help="Data file to be used for uncertainty calibration.",
    )
    predict_gnn_files_args.add_argument(
        "--calibration_features_path",
        type=str,
        nargs="+",
        help="Feature data file to be used with the uncertainty calibration dataset.",
    )
    predict_gnn_files_args.add_argument("--calibration_phase_features_path", type=str, help="")
    predict_gnn_files_args.add_argument(
        "--calibration_atom_descriptors_path",
        type=str,
        help="Extra atom descriptors file.",
    )
    predict_gnn_files_args.add_argument(
        "--calibration_bond_descriptors_path",
        type=str,
        help="Extra bond descriptors file. Path to the extra bond descriptors that will be used as bond features to "
        "featurize a given molecule.",
    )

    predict_gnn_general_args.add_argument(
        "--drop_extra_columns",
        action="store_true",
        help="Keep only SMILES and new prediction columns in the test data files.",
    )

    predict_gnn_uncertainty_args.add_argument(
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
            "spectra_roundrobin",
            "dirichlet",
        ],
        help="The method of calculating uncertainty.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--calibration_method",
        type=str,
        nargs="+",
        choices=[
            "zscaling",
            "tscaling",
            "zelikman_interval",
            "mve_weighting",
            "platt",
            "isotonic",
        ],
        help="Methods used for calibrating the uncertainty.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--individual_ensemble_predictions",
        action="store_true",
        default=False,
        help="Save individual ensemble predictions.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--evaluation_methods",
        type=str,
        nargs="+",
        help="Methods used for evaluating the uncertainty performance. Only used if the test data provided includes "
        "targets. Available methods are [nll, miscalibration_area, ence, spearman] or any available "
        "classification or multiclass metric.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--evaluation_scores_path",
        type=str,
        help="Location to save the results of uncertainty evaluations.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--uncertainty_dropout_p",
        type=float,
        default=0.1,
        help="The probability to use for Monte Carlo dropout uncertainty estimation.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--dropout_sampling_size",
        type=int,
        default=10,
        help="The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout "
        "used during training.",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--calibration_interval_percentile",
        type=float,
        default=95,
        help="Percentile used in calibration methods. Must be in the range (1,100).",
    )
    predict_gnn_uncertainty_args.add_argument(
        "--regression_calibrator_metric",
        type=str,
        choices=["stdev", "interval"],
        help="Regression calibrator output metric. Regression calibrators can output either a stdev or an inverval.",
    )


def predictdmpnn(args: Namespace) -> None:
    """
    Predict the values using a trained D-MPNN model with the given options.
    """
    createLogger("predictgnn.log")
    opts = PredictGnnOptions(**vars(args))
    chemprop.train.make_predictions(args=opts)
