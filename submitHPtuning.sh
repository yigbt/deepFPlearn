#!/bin/bash
 
#$ -N Hyperparametertuning
#$ -S /bin/bash
 
#$ -l h_rt=168:00:00
#$ -l h_vmem=8G
#$ -p smp 14-28
 
#$ -o /work/$USER/$JOB_NAME-$JOB_ID.log
#$ -j y
 
ml anaconda/5/5.0.1

source activate rdkit2019
 
python /home/hertelj/git-hertelj/code/2019_deepFPlearn/TrainTestInSingleFile.py
