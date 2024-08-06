import sys
from argparse import Namespace

from dfpl.parse import parse_dfpl
from dfpl.convert import convert
from dfpl.interpretgnn import interpretdmpnn
from dfpl.predictgnn import predictdmpnn
from dfpl.train import train
from dfpl.predict import predict
from dfpl.traingnn import traindmpnn


def run_dfpl(args: Namespace):
    subprogram_name = args.method

    # The ".method" attribute is added by the parser,
    # specifying which subprogram was chosen by the user.
    # However, the subprograms don't expect the ".method" attribute,
    # so we remove it here before calling the subprogram.
    del args.method

    {
        "traingnn": traindmpnn,
        "predictgnn": predictdmpnn,
        "interpretgnn": interpretdmpnn,
        "train": train,
        "predict": predict,
        "convert": convert
    }[subprogram_name](args)


def main():
    args = parse_dfpl(*sys.argv[1:])

    # dynamic import after parsing was successful (to allow for faster CLI feedback)
    run_dfpl(args)


if __name__ == "__main__":
    main()
