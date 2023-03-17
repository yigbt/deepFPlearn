import logging
import pathlib
from os import path

import dfpl.autoencoder as ac
import dfpl.fingerprint as fp
import dfpl.options as opt
import dfpl.predictions as p

project_directory = pathlib.Path(__file__).parent.absolute()
test_predict_args = opt.PredictOptions(
    inputFile=f"{project_directory}/data/Sun_etal_dataset.cids.predictionSet.csv",
    outputDir=f"{project_directory}/validation/case_01/results/",
    acFile=f"{project_directory}/validation/case_01/results/Sun_etal_dataset.AC.encoder.weights.hdf5",
    model=f"{project_directory}/validation/case_01/results/AR_compressed-True.full.FNN-.model.hdf5",
    target="AR",
    fpSize=2048,
    type="smiles",
    fpType="topological"
)


def test_predictions():
    opts = test_predict_args

    logging.basicConfig(format="DFPL-%(levelname)s: %(message)s", level=logging.INFO)
    logging.info(f"Predicting compounds in the input file {opts.inputFile} for association with target {opts.target}")

    df = fp.importDataFile(opts.inputFile, import_function=fp.importSmilesCSV, fp_size=opts.fpSize)

    use_compressed = False
    if opts.acFile:
        use_compressed = True
        # load trained model for autoencoder
        (_, encoder) = ac.define_ac_model(input_size=opts.fpSize, encoding_dim=opts.encFPSize)
        encoder.load_weights(opts.acFile)
        # compress the fingerprints using the autoencoder
        df = ac.compress_fingerprints(df, encoder)

    # predict
    df2 = p.predict_values(df=df,
                           opts=opts,
                           use_compressed=use_compressed)

    names_columns = [c for c in df2.columns if c not in ['fp', 'fpcompressed']]

    output_file = path.join(opts.outputDir,
                            path.basename(path.splitext(opts.inputFile)[0]) + ".predictions.csv")
    df2[names_columns].to_csv(path_or_buf=output_file)

    logging.info(f"Predictions done.\nResults written to '{output_file}'.")
