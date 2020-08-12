# deepFPlearn

Link molecular structures of chemicals (in form of topological
fingerprints) with multiple targets.

## Usage

You can either generate your own singularity container from the
provided configuration file or generate a single conda environment to
use the `dfpl` package.

### Singularity container

..

### Use conda environment outside the singularity container

To use this tool outside of the Singularity container first create the
respective conda environment with one of these options:

1. Create the conda env from scratch

    From within the `deepFPlearn` directory, you can create the conda environment with the provided
    yaml file that contains all information and necessary packages

    `conda create -n rdkit2019 -f scripts/conda_env.rdkit2019.yml`

2. Activate the `rdkit2019` environment with

    `conda activate rdkit2019`

3. Install the local `dfpl` package by calling

    `conda develop dfpl`
    
Now, you have several options to work with the `dfpl` package:

- You can the package, providing commandline arguments for training or prediction.
  An easy way is to specify all options in a JSON file and calling 
  
  `python -m dfpl -f path/to/file.json`
  
  See, e.g. the JSON files under `validation/case_XX`
- You can load the `dfpl` package in your Python console start
  the training/prediction functions yourself by providing instances
  of `dfpl.options.TrainingOptions` or `dfpl.options.PredictOptions`.
- You can create a run-configuration in PyCharm using the `dfpl/__main__.py`
  script and providing the commandline arguments there.

**TODO: Fix things below**

To see how `deepFPlearn` is to be used call:
`python deepFPlearn -h`

```usage: deepFPlearn [-h] {train,predict} ...

positional arguments:
  {train,predict}  Sub programs of deepFPlearn
    train          Train new models with your data
    predict        Predict your data with existing models

optional arguments:
  -h, --help       show this help message and exit
```

For subcommand specific options, call:
`python deepFPlearn predict -h`

```
usage: deepFPlearn predict [-h] -i FILE --ACmodel FILE --model FILE [-o FILE]
                           [-t STR] [-k STR]

optional arguments:
  -h, --help      show this help message and exit
  -i FILE         The file containin the data for the prediction. It is
                  incomma separated CSV format. The column named 'smiles' or
                  'fp'contains the field to be predicted. Please adjust the
                  type that should be predicted (fp or smile) with -t option
                  appropriately.An optional column 'id' is used to assign the
                  outcomes to theoriginal identifieres. If this column is
                  missing, the results arenumbered in the order of their
                  appearance in the input file.A header is expected and
                  respective column names are used.
  --ACmodel FILE  The autoencoder model weights
  --model FILE    The predictor model weights
  -o FILE         Output file name. It containes a comma separated list of
                  predictions for each input row, for all targets. If the file
                  'id'was given in the input, respective IDs are used,
                  otherwise therows of output are numbered and provided in the
                  order of occurencein the input file.
  -t STR          Type of the chemical representation. Choices: 'fp',
                  'smiles'.
  -k STR          The type of fingerprint to be generated/used in input file.
```

Or for training:

`python deepFPlearn train -h`

```
usage: deepFPlearn train [-h] -i FILE [-o FILE] [-t STR] [-k STR] [-s S] [-a]
                         [-d INT] [-e INT] [-p FILE] [-m] [-l INT] [-K INT]
                         [-v INT]

optional arguments:
  -h, --help  show this help message and exit
  -i FILE     The file containin the data for training in (unquoted) comma
              separated CSV format. First column contain the feature string in
              form of a fingerprint or a SMILES (see -t option). The remaining
              columns contain the outcome(s) (Y matrix). A header is expected
              and respective column names are used to refer to outcome(s)
              (target(s)).
  -o FILE     Prefix of output file name. Trained model(s) and respective
              stats will be returned in 2 output files with this prefix.
              Default: prefix of input file name.
  -t STR      Type of the chemical representation. Choices: 'fp', 'smiles'.
  -k STR      The type of fingerprint to be generated/used in input file.
  -s S        Size of fingerprint that should be generated.
  -a          Use autoencoder to reduce dimensionality of fingerprint.
              Default: not set.
  -d INT      Size of encoded fingerprint (z-layer of autoencoder).
  -e INT      Number of epochs that should be trained
  -p FILE     CSV file containing the parameters for the epochs per target
              model.The target abbreviation should be the same as in the input
              file andthe columns/parameters are:
              target,batch_size,epochs,optimizer,activation.Note that values
              in from file overwrite -e option!
  -m          Train multi-label classification model in addition to the
              individual models.
  -l INT      Fraction of the dataset that should be used for testing. Value
              in [0,1].
  -K INT      K that is used for K-fold cross validation in the training
              procedure.
  -v INT      Verbosity level. O: No additional output, 1: Some additional
              output, 2: full additional output
```

# Please note that:

This work is still in progress. So, if you fork or use it, please contact me first: jana.schor@ufz.de
