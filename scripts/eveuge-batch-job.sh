#!/bin/bash -l

#$ -N dfpl
#$ -S /bin/bash

# requesting resources
#$ -l gpu=1
#$ -l h_rt=168:00:00
#$ -l h_vmem=8G

# Standard output and error:
#$ -o /work/$USER/$JOB_NAME-$JOB_ID.out
#$ -e /work/$USER/$JOB_NAME-$JOB_ID.err
#$ -j y

# Initial working directory:
#$ -wd /work/$USER/deepFPlearn

# mail notification (b-egin, e-nd, a-bort)
#$ -m bea
#$ -M jana.schor@ufz.de

module purge
# use a CUDA-enabled (EasyBuild) toolchain
module load gcccuda
module load fosscuda

module load anaconda/5/5.0.1
source /global/apps/bioinf/tools/anaconda/5/5.0.1/etc/profile.d/conda.sh
conda activate rdkit2019
conda develop dfpl

# Run the program:
#scripts/run-all-cases.sh > eveuge_dfpl_stdout.txt
python -m dfpl train -f "validation/case_02/train.json"