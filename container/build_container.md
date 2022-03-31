# Building the docker container for DFPL

The docker container is built automatically for the latest version of DFPL, and
information how to download/use them can be found in the main `README.md` of this repository.
This guide is only for those who want to build the container locally.

To build the container, you need docker installed on your machine.

```shell
docker build -t TAG -f container/Dockerfile ./ 
```
Replace `TAG` by a tag of your choice and run the container with 

```shell
docker run TAG dfpl DFPL_ARGS
```

replacing `DFPL_ARGS` with the arguments for dfpl.

# Building the singularity container for DFPL

If you want to be able to run the container as a normal user, you need to configure it appropriately when 
[building Singularity from source](https://sylabs.io/guides/3.4/user-guide/installation.html).
In the configuration step, you should provide

```shell script
$ ./mconfig --without-suid --prefix=/home/patrick/build
```

where the last `prefix` option defines where Singularity will be installed.
All other steps are as pointed out in the documentation linked above.

In order to obtain a singularity container the docker container needs to be built first.
Then run:

```
singularity build FILENAME.sif  docker-daemon://TAG
```