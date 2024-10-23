import argparse
import json
from argparse import Namespace
from pathlib import Path

from dfpl.convert import parseInputConvert
from dfpl.interpretgnn import parseInterpretGnn
from dfpl.predictgnn import parsePredictGnn
from dfpl.traingnn import parseTrainGnn
from dfpl.train import parseInputTrain
from dfpl.predict import parseInputPredict


def parse_dfpl(*cli_args: str, **kwargs) -> Namespace:
    """
    Main function that runs training/prediction defined by command line arguments
    """
    parser = argparse.ArgumentParser(prog="deepFPlearn")

    # all subcommands might have common arguments
    # -> use this parent parser to register them
    common_args = argparse.ArgumentParser(add_help=False)

    subparsers = parser.add_subparsers(
        dest="method",  # this allows to check the name of the subparser that was invoked via the .method attribute
        help="Sub programs of deepFPlearn")

    parseTrainGnn(subparsers.add_parser("traingnn",
                                        help="Train new GNN models with your data",
                                        parents=[common_args]))
    parsePredictGnn(subparsers.add_parser("predictgnn",
                                          help="Predict with your GNN models",
                                          parents=[common_args]))
    parseInterpretGnn(subparsers.add_parser("interpretgnn",
                                            help="Interpret your GNN models",
                                            parents=[common_args]))
    parseInputTrain(subparsers.add_parser("train",
                                          help="Train new models with your data",
                                          parents=[common_args]))
    parseInputPredict(subparsers.add_parser("predict",
                                            help="Predict your data with existing models",
                                            parents=[common_args]))
    parseInputConvert(subparsers.add_parser("convert",
                                            help="Convert known data files to pickle serialization files",
                                            parents=[common_args]))

    if len(cli_args) == 0:
        parser.error("Need at least one argument")

    # handle the --configFile argument with a separate parser
    # and extract additional args from config file
    config_arg_parser = argparse.ArgumentParser(add_help=False)
    config_arg_parser.add_argument(
        "-f",
        "--configFile",
        type=str,
        metavar="FILE",
        help="Path to a JSON file containing arguments. CLI arguments will override these.",
        default=None,
    )
    # extract AFTER the first CLI argument (name of the method / sub program)
    cli_args = [cli_args[0],
                *extract_config_args(list(cli_args)[1:], config_arg_parser),
                *dict_to_cli_args(kwargs)]

    return parser.parse_args(cli_args)


def dict_to_cli_args(d: dict) -> list[str]:
    """
    Takes a dict (e.g. parsed from a JSON file)
    and converts it into a list of arguments
    that argparse can interpret as CLI arguments.

    Example:
        {
            "inputFile": "/home/hello world.txt",
            "myBool": true,
            "myInt": 123,
            "myList": [1, "abc", 3]
        }
        becomes
        [
            "--inputFile", "/home/hello world.txt",
            "--myBool",
            "--myInt", "123",
            "--myList", "1", "abc", "3"
        ]
    """
    args = []
    for key, value in d.items():
        key = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(key)
        elif isinstance(value, list):
            args.append(key)
            args.extend(map(str, value))  # simply append all list elements as strings
        else:
            # regular case
            args.append(key)
            args.append(str(value))
    return args


def extract_config_args(args: list[str],
                        config_arg_parser: argparse.ArgumentParser) -> list[str]:
    """
    Looks for the "configFile" argument,
    extracts the CLI arguments from the file and puts them in front.
    This is done recursively.

    Args:
        args: List of raw CLI arguments (might contain --configFile ...)
        config_arg_parser: A parser that only has a single configFile argument registered.

    Returns:
        A list of CLI arguments with all arguments from the config files extracted.
        This list is guaranteed to no longer contain any --configFile arguments.
    """
    config_arg_result, remaining_args = config_arg_parser.parse_known_args(args)
    if config_arg_result.configFile:
        json_file = Path(config_arg_result.configFile)
        if json_file.exists() and json_file.is_file():
            with json_file.open() as f:
                d = json.load(f)
            extracted_args = dict_to_cli_args(d)

            # insert extracted arguments BEFORE the other CLI arguments
            # (give original CLI arguments priority over the extracted ones)
            remaining_args[0:] = extract_config_args(extracted_args, config_arg_parser)  # recursive extraction
        else:
            raise ValueError("Could not find JSON config file", config_arg_result.configFile)
    return remaining_args
