#!/bin/bash

#$ -N dfpl_train_gpu_noAC
#$ -S /bin/bash
#$ -l h_rt=10:00:00
#$ -l h_vmem=30G	# singularity requires a fair bit of memory
#$ -l gpu=1	# request GPUs here!
#$ -binding linear:1

# output files
#$ -o /work/$USER/$JOB_NAME.out
#$ -e /work/$USER/$JOB_NAME.err

# Train on Sun et al data
singularity run --nv /global/apps/deepFPlearn/deepfplearn.sif train -i /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/kfoldCV_GPU/noAC/ -t 'smiles' -k 'topological' -s 2048 -e 2000 -l 0.2 -K 5 -v 2

# Train on BindingDB data
singularity run --nv /global/apps/deepFPlearn/deepfplearn-sc.sif train -i /data/bioinf/projects/data/2020_deepFPlearn/dataSources/BindingDB/data/07_BindingDB.trainingSet.csv -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/kfoldCV_GPU/noAC/ -t 'smiles' -k 'topological' -s 2048 -e 2000 -l 0.2 -K 5 -v 2 -m

# Train on combinded Sun-BindingDB data set
singularity run --nv /global/apps/deepFPlearn/deepfplearn-sc.sif train -i /data/bioinf/projects/data/2020_deepFPlearn/dataSources/combinedBDBSund/01_combinedSUN-BDB.dataset.4training.csv -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/kfoldCV_GPU/noAC/combinedBDBSUN/ -t 'smiles' -k 'topological' -s 2048 -e 2000 -l 0.2 -K 5 -v 2 -m

#Your job 6639483 ("dfpl_train_gpu_noAC") has been submitted
# 09.07.2020 13:17
