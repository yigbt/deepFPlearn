#!/bin/bash

#$ -N dfpl_train_gpu
#$ -S /bin/bash
#$ -l h_rt=00:20:00
#$ -l h_vmem=20G	# singularity requires a fair bit of memory
#$ -l gpu=1	# request GPUs here!
#$ -binding linear:1

# output files
#$ -o /work/$USER/$JOB_NAME.out
#$ -e /work/$USER/$JOB_NAME.err

singularity run --nv /global/apps/deepFPlearn/deepfplearn.sif train -i /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/kfoldCV/ -t 'smiles' -k 'topological' -s 2048 -a -d 256 -e 2000 -l 0.2 -K 5 -v 2
