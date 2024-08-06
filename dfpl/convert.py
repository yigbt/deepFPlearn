from __future__ import annotations

import argparse
import logging
from argparse import Namespace
from os import path

from dfpl import fingerprint as fp
from dfpl.utils import makePathAbsolute, createLogger


def parseInputConvert(parser_convert: argparse.ArgumentParser) -> None:
    """
    Parse the input arguments.

    :return: A namespace object built up from attributes parsed out of the cmd line.
    """
    parser_convert.add_argument(
        "-f",
        metavar="FILE",
        type=str,
        help="Input directory where your CSV/TSV files are stored.",
        required=True,
        default="",
    )


def convert(args: Namespace):
    directory = makePathAbsolute(args.f)
    if path.isdir(directory):
        createLogger(path.join(directory, "convert.log"))
        logging.info(f"Convert all data files in {directory}")
        fp.convert_all(directory)
    else:
        raise ValueError("Input directory is not a directory")
