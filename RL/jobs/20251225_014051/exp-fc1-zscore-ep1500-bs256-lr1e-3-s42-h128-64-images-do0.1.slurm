#!/bin/bash
#SBATCH --job-name=exp-fc1-zscore-ep1500-bs256-lr1e-3-s42-h128-64-images-do0.1
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251225_014051/exp-fc1-zscore-ep1500-bs256-lr1e-3-s42-h128-64-images-do0.1.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251225_014051/exp-fc1-zscore-ep1500-bs256-lr1e-3-s42-h128-64-images-do0.1.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:30:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_job.py --datapath data/dataset100K25M_LA_rand.pkl --model_type fc1 --norm zscore --epochs 1500 --batch_size 256 --lr 1e-3 --val_split 0.2 --seed 42 --train_type images --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251225_014051/exp-fc1-zscore-ep1500-bs256-lr1e-3-s42-h128-64-images-do0.1 --weight_decay 0.0 --mag_cost 1.0 --fc1_hidden 128 64 --fc1_activation leaky_relu --fc1_dropout 0.1 --fc1_final_activation none --cpu_only
