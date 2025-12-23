#!/bin/bash
#SBATCH --job-name=exp-fc1-zscore-ep2000-bs2048-lr1e-3-s42-h128-64-images-wd0.0005
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251222_203559/exp-fc1-zscore-ep2000-bs2048-lr1e-3-s42-h128-64-images-wd0.0005.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251222_203559/exp-fc1-zscore-ep2000-bs2048-lr1e-3-s42-h128-64-images-wd0.0005.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:30:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_job.py --datapath data/dataset100K25M_LA.pkl --model_type fc1 --norm zscore --epochs 2000 --batch_size 2048 --lr 1e-3 --val_split 0.2 --seed 42 --train_type images --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251222_203559/exp-fc1-zscore-ep2000-bs2048-lr1e-3-s42-h128-64-images-wd0.0005 --weight_decay 0.0005 --mag_cost 1.0 --fc1_hidden 128 64 --fc1_activation leaky_relu --fc1_dropout 0.0 --fc1_encoder_enabled --fc1_filter_sizes 3 3 3 --fc1_filter_channels 8 16 32 --fc1_final_embedding_size 16 --fc1_final_embedding_channels 32 --fc1_final_activation none --cpu_only
