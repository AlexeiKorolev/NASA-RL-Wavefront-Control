#!/bin/bash
#SBATCH --job-name=exp-fc1-zscore-ep5000-bs262144-lr1e-3-s42-h512-256-128-slopes
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_191750/exp-fc1-zscore-ep5000-bs262144-lr1e-3-s42-h512-256-128-slopes.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_191750/exp-fc1-zscore-ep5000-bs262144-lr1e-3-s42-h512-256-128-slopes.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_job.py --datapath data/dataset220K25M.pkl --model_type fc1 --norm zscore --epochs 5000 --batch_size 262144 --lr 1e-3 --val_split 0.2 --seed 42 --train_type slopes --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_191750/exp-fc1-zscore-ep5000-bs262144-lr1e-3-s42-h512-256-128-slopes --fc1_hidden 512 256 128
