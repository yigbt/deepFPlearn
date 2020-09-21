# Building the Singularity container for DFPL

To build the container, you need [Singularity](https://sylabs.io/guides/3.4/user-guide/installation.html) installed
on your machine.
If you want to be able to run the container as a normal user, you need to configure it appropriately when building
Singularity from source.
In the configuration step, you should provide

```shell script
$ ./mconfig --without-suid --prefix=/home/patrick/build
```

where the last `prefix` option defines where Singularity will be installed.
All other steps are as pointed out in the documentation linked above.

Building the container using the provided `conda_rdkit2019.def` definition file requires only a few steps:

1. Download the `tensorflow` base container which gives us many requirements for free
2. Build a package from the DFPL sources, because DFPL will be installed inside the container
3. (Optionally) Make some adjustments to the `.def` file
3. Run the container build command

## 1. Download Tensorflow base container

Actually, it is not strictly necessary to download the Tensorflow base-container.
However, for testing this speeds up the process since you only download it once and can use your local copy everytime
you need to rebuild your container.
Additionally, the `.def` file is set up to use the `tensorflow_latest-gpu.sif` file from the same directory.
Therefore, from within the `singularity_container` directory, execute
```shell script
singularity pull docker://tensorflow/tensorflow:latest-gpu
```

This will save the downloaded `.sif` file.

## 2. Building the DFPL package

You need Python 3 for this with things like `setuptools` installed.
The easiest way is to do this from within the conda `rdkit2019` environment if you have created it locally
From the `deepFPlearn` directory, the following command will pack DFPL and write the output to the `singularity_container`
directory.

```shell script
python setup.py bdist_wheel -d singularity_container
```

After this, you should have a `dfpl-XX.XX-py3-none-any.whl` file in the `singularity_container` directory.

## 3. Make some adjustments to the `.def` file

You can adjust this file to your liking, however, one important step is to verify that the file-name of DFPL package
at the top of the `.def` file matches the `.whl` file you created in step 2.
Also, the definitions for the conda `rdkit2019` environment are in the file `environment.yml`.

## 4. Run the container build command

Building the container needs to be done with sudo rights.
From within the `singularity_container` directory, run the following commands:

```shell script
SING=$(command -v singularity)
sudo $SING build conda_rdkit2019.sif conda_rdkit2019.def
```

# Using the DFPL container

If you want to use your GPU, you need to provide the `--nv` commandline option when running the container.
In `run` mode, the container will automatically activate the `rdkit2019` conda environment for you, and you can
easily run the DFPL package:

```shell script
singularity run --nv singularity_container/conda_rdkit2019.sif "python -m dfpl convert -f \"data\""
```

or you can run all cases using

```shell script
singularity run --nv singularity_container/conda_rdkit2019.sif ". ./scripts/run-all-cases.sh"
```

It's also possible to get an interactive shell into the container

```shell script
singularity shell --nv singularity_container/conda_rdkit2019.sif
```

To activate the `rdkit2019` conda environment, you can run the following inside the container shell

```shell script
source /.start_dfpl_env 
```