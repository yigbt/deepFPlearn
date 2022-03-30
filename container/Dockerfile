FROM continuumio/miniconda3 AS build

# Install the package as normal:
COPY container/environment.yml .
RUN conda env create -f environment.yml

# Install conda-pack:
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n dfpl_env -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack


# The runtime-stage image 
FROM tensorflow/tensorflow:latest-gpu AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

# copy dfpl sources for installation
# unfortunatelly COPY setup.py README.md dfpl/ /deepFPlearn/
# does not work sind it just copies the files in dfpl
COPY ./ /deepFPlearn/

# install dfpl
RUN source /venv/bin/activate && python -m pip install --no-deps /deepFPlearn

# The code to run when container is started. 
ENTRYPOINT ["/deepFPlearn/container/entrypoint.sh"]
# docker run dfpl:latest dfpl DFPL_ARGUMENTS # to run dfpl
# docker run -it dfpl:latest bash # and an interactive shell
# 
# if using the container like an executable, ie 
# `docker run dfpl:latest DFPL_ARGUMENTS`
# then change the last line in entrypoint.sh to just `dfpl $@`
# this does not easily allow to obtain an interactive shell
# maybe with --entrypoint?

LABEL author="Jana Schor, Patrick Scheibe, Matthias Bernt"
LABEL version="1.0"