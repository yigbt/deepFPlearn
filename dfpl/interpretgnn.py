from __future__ import annotations

import argparse
from argparse import Namespace
from typing import List

import chemprop
from chemprop.args import InterpretArgs

from dfpl.utils import createLogger


class InterpretGNNoptions(InterpretArgs):
    """
    Dataclass to hold all options used for training the graph models
    """

    # configFile: str = "./example/interpret.json"
    data_path: str = "./example/data/smiles.csv"
    batch_size: int = 500
    c_puct: float = 10.0
    max_atoms: int = 20
    min_atoms: int = 8
    prop_delta: float = 0.5
    property_id: List[int] = None
    rollout: int = 20


def parseInterpretGnn(parser_interpret_gnn: argparse.ArgumentParser) -> None:
    interpret_gnn_files_args = parser_interpret_gnn.add_argument_group("Files")
    interpret_gnn_interpret_args = parser_interpret_gnn.add_argument_group("Interpretation Configuration")
    interpret_gnn_files_args.add_argument(
        "--preds_path",
        type=str,
        metavar="FILE",
        help="Path to CSV file where predictions will be saved",
        default="",
    )
    interpret_gnn_files_args.add_argument(
        "--checkpoint_dir",
        type=str,
        metavar="DIR",
        help="Directory from which to load model checkpoints"
             "(walks directory and ensembles all models that are found)",
        default="./ckpt",
    )
    interpret_gnn_files_args.add_argument(
        "--checkpoint_path",
        type=str,
        metavar="DIR",
        help="Path to model checkpoint (.pt file)",
    )
    interpret_gnn_files_args.add_argument(
        "--data_path",
        type=str,
        metavar="FILE",
        help="Path to CSV file  for which predictions will be made",
    )
    interpret_gnn_interpret_args.add_argument(
        "--max_atoms",
        type=int,
        metavar="INT",
        help="Maximum number of atoms to use for interpretation",
    )

    interpret_gnn_interpret_args.add_argument(
        "--min_atoms",
        type=int,
        metavar="INT",
        help="Minimum number of atoms to use for interpretation",
    )

    interpret_gnn_interpret_args.add_argument(
        "--prop_delta",
        type=float,
        metavar="FLOAT",
        help="The minimum change in the property of interest that is considered significant",
    )
    interpret_gnn_interpret_args.add_argument(
        "--property_id",
        type=int,
        metavar="INT",
        help="The index of the property of interest",
    )
    # write the argument for rollouts
    interpret_gnn_interpret_args.add_argument(
        "--rollout",
        type=int,
        metavar="INT",
        help="The number of rollouts to use for interpretation",
    )


def interpretdmpnn(args: Namespace) -> None:
    """
    Interpret the predictions of a trained D-MPNN model with the given options.
    """
    createLogger("interpretgnn.log")
    opts = InterpretGNNoptions(**vars(args))
    chemprop.interpret.interpret(args=opts, save_to_csv=True)
