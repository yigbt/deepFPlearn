import logging
import pathlib

import dfpl.__main__ as main
import dfpl.options as opt
import dfpl.utils as utils

project_directory = pathlib.Path(__file__).parent.absolute()
example_directory = project_directory.parent / "example"
test_train_args = opt.GnnOptions(
    configFile=utils.makePathAbsolute(f"{example_directory}/traingnn.json"),
    save_dir=utils.makePathAbsolute(f"{project_directory}/output"),
)


def test_traindmpnn(opts: opt.GnnOptions) -> None:
    print("Running traindmpnn test...")
    logging.basicConfig(
        format="DFPL-{levelname}: {message}", style="{", level=logging.INFO
    )
    logging.info("Adding fingerprint to dataset")

    main.traindmpnn(opts)

    print("Training DMPNN...")

    print("traindmpnn test complete.")


if __name__ == "__main__":
    test_traindmpnn(test_train_args)
