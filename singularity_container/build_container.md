# Building the Singularity container for DFPL

The Singularity container is built automatically for the latest version of DFPL, and
information how to download/use them can be found in the main `README.md` of this repository.
This guide is only for those who want to build the container locally.

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

Building the container using the provided `dfpl.def` definition file requires only a few steps:

1. (Optionally) Make some adjustments to the `.def` file
2. Run the container build command

## 1. Make some adjustments to the `.def` file

You can adjust this file to your liking, e.g. adjust the conda environment that build inside the container
and which is defined in the file `environment.yml`.

## 2. Run the container build command

Building the container needs to be done with sudo rights.
From within the `singularity_container` directory, run the following commands:

```shell script
SING=$(command -v singularity)
sudo $SING build dfpl.sif dfpl.def
```

**Note:** The container will have a copy of the whole `deepFPlearn` directory in it.
Therefore, ensure that you remove old `.sif` files or any other large data-files that you
don't want to include in the container itself. During the automatic building, the `deepFPlearn`
repository is cloned using

```shell
git clone --depth 1 https://github.com/yigbt/deepFPlearn.git
```

to keep the container size at a minimum.

# Using the DFPL container

If you want to use your GPU, you need to provide the `--nv` commandline option when running the container.
In `run` mode, the container will automatically activate the `dfpl_env` conda environment for you, and you can
easily run the DFPL package:

```shell script
singularity run --nv singularity_container/dfpl.sif "python -m dfpl convert -f \"data\""
```

or you can run all cases using

```shell script
singularity run --nv singularity_container/dfpl.sif ". ./scripts/run-all-cases.sh"
```

It's also possible to get an interactive shell into the container

```shell script
singularity shell --nv singularity_container/dfpl.sif
```

To activate the `dfpl_env` conda environment, you can run the following inside the container shell

```shell script
source /.start_dfpl_env 
```