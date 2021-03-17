#!/usr/bin/env bash

# This script needs to be run from the deepFPlearn directory.
# Importantly, the conda environment needs to be set up and actived! For certain machines/HPC,
# we have a batch-job that does exactly that and then calls this file

#D="data/"; if [ -d $D ]; then python -m dfpl convert -f $D; fi
#D="data/MoleculeNet/Biophysics/"; if [ -d $D ]; then python -m dfpl convert -f $D; fi

#F="validation/case_MUV/train_AC-generic.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
#F="validation/case_MUV/train_AC-specific.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
#
#F="validation/case_Tox21/train_AC-generic.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
#F= "validation/case_Tox21/train_AC-specific.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
#
#F="validation/case_HIV/train_AC-generic.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
#F="validation/case_HIV/train_AC-specific.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi

F="validation/case_BACE/train_AC-generic.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F= "validation/case_BACE/train_AC-specific.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi

F="validation/case_sider/train_AC-generic.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F= "validation/case_sider/train_AC-specific.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi

F="validation/case_PCBA/train_AC-generic.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F= "validation/case_PCBA/train_AC-specific.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi

