# Deep Fingerprint Learn (DFPL)

Link molecular structures of chemicals (in form of topological fingerprints) with multiple targets.

## Setting up Python environment

The DFPL package requires a particular Python environment to work properly.
It consists of a recent Python interpreter and packages for data-science and neural networks.
The exact dependencies can be found in the
[`requirements.txt`](requirements.txt) (which is used when installing the package with pip)
and [`environment.yml`](environment.yml) (for installtion with conda).

You have several ways to provide the correct environment to run code from the DFPL package.

1. Use the automatically built docker/Singularity containers
2. Build your own container [following the steps here](container/build_container.md)
3. Setup a python virtual environment
4. Set up a conda environment install the requirements via conda and the DFPL package via pip

In the following, you find details for option 1., 3., and 4.

### Docker container

You need docker installed on you machine. 

In order to run DFPL use the following command line 

```shell
docker run --gpus GPU_REQUEST registry.hzdr.de/department-computational-biology/deepfplearn/deepfplearn:TAG dfpl DFPL_ARGS
```

where you replace

- `TAG` by the version you want to use or `latest` if you want to use latest available version)
- You can see available tags here https://gitlab.hzdr.de/department-computational-biology/deepfplearn/container_registry/5827.
  In general a container should be available for each released version of DFPL.
- `GPU_REQUEST` by the GPUs you want to use or `all` if all GPUs should be used (remove `--gpus GPU_REQUEST` if only the CPU should)
- `DFPL_ARGS` by arguments that should be passed to DFPL (use `--help` to see available options)

In order to get an interactive bash shell in the container use:

```shell
docker run -it --gpus GPU_REQUEST registry.hzdr.de/department-computational-biology/deepfplearn/deepfplearn:TAG bash
```

### Singularity container

You need Singularity installed on your machine. You can download a container with

```shell
singularity pull dfpl.TAG.sif docker://registry.hzdr.de/department-computational-biology/deepfplearn/deepfplearn:TAG
```

- replace `TAG` by the version you want to use or `latest` if you want to use latest available version)
- You can see available tags here https://gitlab.hzdr.de/department-computational-biology/deepfplearn/container_registry/5827.
  In general a container should be available for each released version of DFPL.


This stores the container as a file `dfpl.TAG.sif` which can be run as follows:

```shell script
singularity run --nv dfpl.TAG.sif dfpl DFPL_ARGS
```
- replace `DFPL_ARGS` by arguments that should be passed to DFPL (use `--help` to see available options)
- omit the `--nv` tag if you don't want to use GPUs 


or you can start a shell script (look at [run-all-cases.sh](scripts/run-all-cases.sh) for an
example)

```shell script
singularity run --nv dfpl.sif ". ./scripts/run-all-cases.sh"
```

It's also possible to get an interactive shell into the container

```shell script
singularity shell --nv dfpl.TAG.sif
```

**Note:** The Singularity container is intended to be used on HPC cluster where your ability to install software might
be limited.
For local testing or development, setting up the conda environment is preferable.

### Set up DFPL in a python virtual environment

From within the `deepFPlearn` directory call

```
virtualenv -p python3 ENV_PATH
. ENV_PATH/bin/activate
pip install ./
```

replace `ENV_PATH` by the directory where the python virtual environment should be created.
If your system has only python3 installed `-p python3` may be removed.

In order to use the environment it needs to be activated with `. ENV_PATH/bin/activate`.

### Set up DFPL in a conda environment

To use this tool outside the Singularity container first create the respective conda environment:

1. Create the conda env from scratch

   From within the `deepFPlearn` directory, you can create the conda environment with the provided yaml file that
   contains all information and necessary packages

   ```shell
   conda env create -f environment.yml
   ```

2. Activate the `dfpl_env` environment with

   ```shell
   conda activate dfpl_env
   ```

3. Install the local `dfpl` package by calling

   ```shell
   pip install --no-deps ./
   ```

## Prepare data

DFPL can calculate fingerprints of chemical structures from SMILES or INCHI representation. Therefore, e.g. CSV
input-files need to contain a `"smiles"` or `"inchi"` which is then used to calculate the fingerprints. There is an
example CSV file in the `tests/directory` directory and when you're training using the DFPL package, it will load the
input files and add fingerprints. You can test the conversion

```python
import dfpl.fingerprint as fp

fp.importDataFile("tests/data/smiles.csv")
```

If you're data is in CSV format, has a header row, and contains a `"smiles"` or an `"inchi"` column, you can use it as
input for training as it is. However, if you're data is in a different format, you can use function in the `fingerprint`
module to import it correctly.

The `tests/data/inchi.tsv` contains data in TSV format without a header row which makes it impossible to identify how to
import it automatically. You can use the `import_function` argument to tell `importDataFile` how it can turn your data
into a Pandas `DataFrame` that contains, e.g. an `"inchi"` column. After that DFPL can calculate and add the
fingerprints to the `DataFrame`

```python
import pandas as pd
import dfpl.fingerprint as fp

data = fp.importDataFile(
    "tests/data/inchi.tsv",
    import_function=(lambda f: pd.read_table(f, names=["toxid", "inchi", "key"]))
)
```

You can store the converted data as a "pickle" file which is a binary representation of the Pandas dataframe and can be
used directly as input file for the DFPL program. The advantage is that the fingerprint calculation needs to be done
only once and loading these files is fast.

```python
data.to_pickle("output/path/file.pkl")
```

Note that the file-extension needs to be `"pkl"` to be identified correctly by DFPL. Also, you might want to look at
the `convert_all` function in the `fingerprint` module that we use to convert different data-files all at once.

## Use training/prediction functions

You have several options to work with the DFPL package. The package can be started as a program on the commandline and
you can provide all necessary information as commandline-parameters. Check

```shell script
dfpl --help
dfpl train --help
dfpl predict --help
```

However, using JSON files that contain all train/predict options an easy way to preserve what was run and you can use
them instead of providing multiple commandline arguments.

```shell script
dfpl train -f path/to/file.json
```

See, e.g. the JSON files under `validation/case_XX` for examples. Also, you can use the following to create template
JSON files for training or prediction

```python
import dfpl.options as opts

train_opts = opts.TrainOptions()
train_opts.saveToFile("train.json")

predict_opts = opts.PredictOptions()
predict_opts.saveToFile("predict_bestER03.json")
```

You can also work with the DFPL package from within an interactive Python session. Load the `dfpl` package in your
Python console and start the training/prediction functions yourself by providing instances
of `dfpl.options.TrainingOptions` or
`dfpl.options.PredictOptions`. You can also use load options from JSON files. Example

```python
import dfpl.__main__ as main
import dfpl.options as opts

o = opts.TrainOptions.fromJson("/path/to/train.json")
main.train(o)
```

Finally, if you like to work in PyCharm, you can also create a run-configuration in PyCharm using the `dfpl/__main__.py`
script and providing the commandline arguments there.

# Please note that:

This work has been submitted in a research paper and is currently under review.
You may read the preprint @bioRxiv: https://doi.org/10.1101/2021.06.24.449697

For questions, comments please contact me: jana.schor@ufz.de
