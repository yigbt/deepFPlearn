#!/bin/bash -l
# Standard output and error:
#SBATCH -o /u/pscheibe/deepFPlearn/tjob.out.%j
#SBATCH -e /u/pscheibe/deepFPlearn/tjob.err.%j
# Initial working directory:
#SBATCH -D /u/pscheibe/deepFPlearn
#
#SBATCH -J dfpl_analysis
#
# Node feature:
#SBATCH --constraint="gpu"
# Specify type and number of GPUs to use:
#   GPU type can be v100 or rtx5000
#SBATCH --gres=gpu:v100:2         # If using both GPUs of a node
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40      # If using both GPUs of a node
#
#SBATCH --mail-type=none
#SBATCH --mail-user=pscheibe@rzg.mpg.de
#
# wall clock limit:
#SBATCH --time=24:00:00
module purge
module load cuda
module load anaconda/3/2020.02

# Run the program:
srun scripts/mpcpf-run.sh > prog.out
