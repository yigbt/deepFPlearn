FROM tensorflow/tensorflow:latest-gpu

# copy dfpl sources for installation
# unfortunatelly COPY setup.py README.md dfpl/ /deepFPlearn/
# does not work sind it just copies the files in dfpl
COPY ./ /deepFPlearn/

# install dfpl
RUN python -m pip install /deepFPlearn && pip install pytest

# The code to run when container is started. 
CMD ["bash"]
# docker run dfpl:latest dfpl DFPL_ARGUMENTS # to run dfpl
# docker run -it dfpl:latest # and an interactive shell
# 
# if using the container like an executable, ie 
# `docker run dfpl:latest DFPL_ARGUMENTS`
# then change the last line in entrypoint.sh to just `dfpl $@`
# this does not easily allow to obtain an interactive shell
# maybe with --entrypoint?

LABEL author="Jana Schor, Patrick Scheibe, Matthias Bernt"
LABEL version="1.0"