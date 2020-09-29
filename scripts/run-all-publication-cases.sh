#!/usr/bin/env bash

# This script needs to be run from the deepFPlearn directory.
# Importantly, the conda environment needs to be set up and actived! For certain machines/HPC,
# we have a batch-job that does exactly that and then calls this file

D="data"; if [ -d $D ] python -m dfpl convert -f $D; fi

F="validation/case_00/train_AC_S.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_00/train_AC_D.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi

F="validation/case_01/train.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_02/train.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_03/train.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi

F="validation/case_07/predict_bestER03_Sdata.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_07/predict_bestER03.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_07/predict_fullER03.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
#F="validation/case_07/predict_bestARext03.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_07/predict_bestED03.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi

F="validation/case_01/train_0p5.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_01/train_0p6.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_01/train_0p7.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_01/train_0p8.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_01/train_0p9.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_01/train_1p0.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi

F="validation/case_02/train_0p5.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_02/train_0p6.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_02/train_0p7.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_02/train_0p8.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_02/train_0p9.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi
F="validation/case_02/train_1p0.json"; if [ -f $F ]; then python -m dfpl train -f $F; fi

