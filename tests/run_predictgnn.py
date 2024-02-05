import logging
import pathlib

from chemprop import train
import chemprop as cp
import dfpl.options as opt
import dfpl.utils as utils

project_directory = pathlib.Path(__file__).parent.absolute()
example_directory = (
    project_directory.parent / "example"
)  # Path to the example directory

test_predict_args = opt.GnnOptions(
    configFile=utils.makePathAbsolute(f"{example_directory}/predictgnn.json"),
    save_dir=utils.makePathAbsolute(f"{project_directory}/output"),
)


def test_predictdmpnn(opts: opt.GnnOptions) -> None:
    print("Running predictdmpnn test...")
    logging.basicConfig(
        format="DFPL-{levelname}: {message}", style="{", level=logging.INFO
    )

    json_arg_path = utils.makePathAbsolute(f"{example_directory}/predictgnn.json")
    arguments = utils.createArgsFromJson(json_arg_path)
    opts = cp.args.PredictArgs().parse_args(arguments)

    train.make_predictions(args=opts)

    print("predictdmpnn test complete.")


if __name__ == "__main__":
    test_predictdmpnn(test_predict_args)
