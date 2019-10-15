#!/bin/bash


/data/conda_envs/rdkit2019/bin/python /home/hertelj/git-hertelj/code/2019_deepFPlearn/deepFPlearn-Train.py -i /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/ -t smiles -k topological -e 10000
