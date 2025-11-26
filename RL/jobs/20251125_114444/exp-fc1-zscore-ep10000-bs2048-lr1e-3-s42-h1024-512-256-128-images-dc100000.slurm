#!/bin/bash
#SBATCH --job-name=exp-fc1-zscore-ep10000-bs2048-lr1e-3-s42-h1024-512-256-128-images-dc100000
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251125_114444/exp-fc1-zscore-ep10000-bs2048-lr1e-3-s42-h1024-512-256-128-images-dc100000.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251125_114444/exp-fc1-zscore-ep10000-bs2048-lr1e-3-s42-h1024-512-256-128-images-dc100000.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_job.py --datapath data/dataset100K25M100N_orth_ood_2e-8.pkl --model_type fc1 --norm zscore --epochs 10000 --batch_size 2048 --lr 1e-3 --val_split 0.2 --seed 42 --train_type images --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251125_114444/exp-fc1-zscore-ep10000-bs2048-lr1e-3-s42-h1024-512-256-128-images-dc100000 --data_cutoff 100000 --fc1_hidden 1024 512 256 128 --cpu_only
