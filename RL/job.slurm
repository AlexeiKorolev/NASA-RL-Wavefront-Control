#!/bin/bash
#SBATCH --job-name=testjob          # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G            # memory per cpu-core (4G is default)
#SBATCH --time=00:01:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /home/ak9088/anaconda3/envs/dmrl2

python /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/environment.py