#!/bin/bash
#SBATCH --job-name=exp-fc1-zscore-ep200-bs256-lr1e-3-s42
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251017_110450/exp-fc1-zscore-ep200-bs256-lr1e-3-s42.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251017_110450/exp-fc1-zscore-ep200-bs256-lr1e-3-s42.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_job.py --datapath data/dataset100K25M.pkl --model_type fc1 --norm zscore --epochs 200 --batch_size 256 --lr 1e-3 --val_split 0.2 --seed 42 --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251017_110450/exp-fc1-zscore-ep200-bs256-lr1e-3-s42
