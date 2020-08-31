#!/bin/bash -l
# Standard output and error:
SBATCH -o ./tjob.out.%j
SBATCH -e ./tjob.err.%j
# Initial working directory:
SBATCH -D ./
#
SBATCH -J dfpl_analysis
#
# Node feature:
SBATCH --constraint="gpu"
# Specify type and number of GPUs to use:
#   GPU type can be v100 or rtx5000
SBATCH --gres=gpu:v100:2         # If using both GPUs of a node
# #SBATCH --gres=gpu:v100:1       # If using only 1 GPU of a shared node
# #SBATCH --mem=92500             # Memory is necessary if using only 1 GPU
#
# Number of nodes and MPI tasks per node:
SBATCH --nodes=1
SBATCH --ntasks-per-node=40      # If using both GPUs of a node
# #SBATCH --ntasks-per-node=20    # If using only 1 GPU of a shared node
#
#SBATCH --mail-type=none
#SBATCH --mail-user=<userid>@rzg.mpg.de
#
# wall clock limit:
#SBATCH --time=24:00:00
module purge
module load cuda
module load anaconda/3/2020.02

# Run the program:
srun scripts/mpcpf-run.sh > prog.out
