#!/bin/bash -l
# Standard output and error:
#SBATCH -o /u/pscheibe/deepFPlearn/mpcdf_job.out.%j
#SBATCH -e /u/pscheibe/deepFPlearn/mpcdf_job.err.%j
#
# Initial working directory:
#SBATCH -D /u/pscheibe/deepFPlearn
#
#SBATCH -J dfpl
#
#SBATCH --partition="gpu"
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:2         # If using both GPUs of a node
#SBATCH --mem=92500
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#
#SBATCH --mail-type=none
#SBATCH --mail-user=pscheibe@rzg.mpg.de
#
#SBATCH --time=24:00:00
module purge
module load cuda

source /u/pscheibe/conda/etc/profile.d/conda.sh
conda activate dfpl_env
conda develop dfpl

# Run the program:
srun scripts/run-all-publication-cases.sh &> mpcdf_dfpl_run.log
