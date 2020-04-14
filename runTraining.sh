#!/bin/bash

python deepFPlearn train -i /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/kfoldCV -t 'smiles' -k 'topological' -s 2048 -a -d 256 -e 2000 -l 0.2 -K 5 -v 2
