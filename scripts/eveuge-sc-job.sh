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

# mail notification (b-egin, e-nd, a-bort)
#$ -m bea
#$ -M jana.schor@ufz.de

# Train case_01
singularity run --nv /global/apps/bioinf/singularity_images/conda_rdkit2019.sif "python -m dfpl train -f /home/hertelj/git-hertelj/code/deepFPlearn/validation/case_02/trainEVE.json"
