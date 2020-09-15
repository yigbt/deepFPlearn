#!/bin/bash -l

#$ -N dfpl
#$ -S /bin/bash

# requesting resources
#$ -l h_rt=168:00:00
#$ -l h_vmem=8G
#$ -l gpu=1
#$ -binding linear:1


# Standard output and error:
#$ -o /work/$USER/$JOB_NAME-$JOB_ID.out
#$ -e /work/$USER/$JOB_NAME-$JOB_ID.err
#$ -j y

# Initial working directory:
#$ -wd /home/hertelj/git-hertelj/code/deepFPlearn/

# mail notification (b-egin, e-nd, a-bort)
#$ -m bea
#$ -M jana.schor@ufz.de

module purge
# use a CUDA-enabled (EasyBuild) toolchain
#module load gcccuda
module load fosscuda

module load Anaconda2/2019.10 
source /software/easybuild-broadwell/software/Anaconda2/2019.10/etc/profile.d/conda.sh
conda activate rdkit2019TF 
#conda develop -u dfpl
conda develop dfpl

# Run the program:
#scripts/run-all-cases.sh > eveuge_dfpl_stdout.txt
python -m dfpl train -f "/home/hertelj/git-hertelj/code/deepFPlearn/validation/case_02/train.json"

#python /home/hertelj/git-hertelj/code/deepFPlearn/doIgetTheGPU.py
