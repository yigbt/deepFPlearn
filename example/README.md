Here you find example code for running `deepFPlearn` in all three modes.

The input for each of these scripts can be found in the `data` folder.
The pre-computed output can be found in the `results_[train,predict,convert]` folder.
Trained models that are used in the prediction mode are stored in the `models` folder.

## Train

**Script to use:** `deepFPlearn_train.sh`

Use this script to train a specific autoencoder with a provided data set, and subsequently train feed forward networks
for the targets in this data set.

This script should be called from the `examples` folder:
```
cd example
./deepFPlearn_train.sh
```
The trained models, i.e. the stored weights for all neurons as `.hdf5` files, history plots and modelscores are 
stored in the `results_train` folder when you run this program.
Pre-computed results can be found in the github release assets.

## Predict

**Script to use:** `deepFPlearn_predict.sh`

Use this script to predict the provided set of compounds using generic feature compression and the best AR model for 
associations to androgen receptor.

This script should be called from the `examples` folder:
```
cd example
./deepFPlearn_predict.sh
```
The compounds are predicted using a *random* model (column `random` in the output) and the *trained* model 
(colum `trained` in the output).

## Convert

**Script to use:** `deepFPlearn_convert.sh`

This mode is used to convert `.csv` or `.tsv` files into `.pkl` files for easy access in Python and to reduce memory on disk.
The `.pkl` files then already contain the binary fingerprints and are ready to use for training oder predicting.

**Note:** Train and Predict modes do not require their inputs to be in .pkl, .csv is also fine but a bit slower.

This script should be called from the `examples` folder:
```
cd example
./deepFPlearn_convert.sh
```
The `.pkl` files are stored in the `examples/data` folder, next to their input files.

If you do this with a custom file of a different file name, you have to edit the code like this:

1. open file dfpl/fingerprint.py
2. search for the conversation_rules
3. add your filename in the same style
