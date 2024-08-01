Here you find example code for running `deepFPlearn` in all three modes.

The input for each of these scripts can be found in the `example/data` folder.

The pre-computed output of the `train` mode can be found in the assets of the release, for the `predict` mode it is 
stored in the respective `example/results_predict` folder.
Trained models that are used in the prediction mode are stored in the `models` folder.

**NOTE**: Before you proceed calling `deepFPlearn` activate the conda environment or use the container as described in the main `README.md` of the repository.

## Train

The training data contains three targets and you may train models for each using the following command.
Training with the configurations from the `example/train.json` file will take approximately 4min on a single CPU.
```
python -m dfpl train -f example/train.json
```
The trained models, training histories and respective plots, as well as the predictions on the test data are stored in the `example/results_train` folder as defined in the `example/train.json` file (you may change this).


## Predict

Use this command to predict the provided data for prediction.
This will take only a few seconds on single CPU.
```
python -m dfpl predict -f example/predict.json
```
The compounds are predicted with the (provided) AR model and results are returned as float number between 0 and 1.

## Convert

This mode is used to convert `.csv` or `.tsv` files into `.pkl` files for easy access in Python and to reduce memory on disk.
The `.pkl` files then already contain the binary fingerprints and are ready to use for training oder predicting.

**Note:** Train and Predict modes do not require their inputs to be in .pkl, .csv is also fine but a bit slower.

```
python -m dfpl convert -f example/data
```
The `.pkl` files are stored in the `examples/data` folder, next to their input files.

If you do this with a custom file of a different file name, you have to edit the code like this:

1. open file dfpl/fingerprint.py
2. search for the conversation_rules
3. add your filename in the same style
