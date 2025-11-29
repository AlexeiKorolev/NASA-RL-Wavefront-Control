#!/bin/bash
#SBATCH --job-name=datagen          # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=24G           # memory per cpu-core (4G is default)
#SBATCH --time=05:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2


python data_generator.py --N 100000 --output data/dataset100K25M_VLA2.pkl --num-modes=25 --basis-type orthogonal --dm-random-noise 2e-6 --num-airy 12 --pixels-per-spacial-res 2 --nudge-magnitude 6e-7