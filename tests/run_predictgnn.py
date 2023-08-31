import logging
import os
import pathlib
import dfpl.options as opt
import dfpl.utils as utils
import pandas as pd
from chemprop import train, args


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
    ignore_elements = [
        "py/object",
        "checkpoint_paths",
        "save_dir",
        "saving_name",
    ]
    arguments, data = utils.createArgsFromJson(
        json_arg_path, ignore_elements, return_json_object=True
    )
    arguments.append("--preds_path")
    arguments.append("")
    save_dir = data.get("save_dir")
    name = data.get("saving_name")

    opts = args.PredictArgs().parse_args(arguments)
    opts.preds_path = os.path.join(save_dir, name)
    df = pd.read_csv(opts.test_path)
    smiles = [[row.smiles] for _, row in df.iterrows()]

    train.make_predictions(args=opts, smiles=smiles)

    print("predictdmpnn test complete.")


if __name__ == "__main__":
    test_predictdmpnn(test_predict_args)
