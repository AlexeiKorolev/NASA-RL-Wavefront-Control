#!/bin/bash
#SBATCH --job-name=exp-fc1-zscore-ep5000-bs2048-lr1e-3-s42-h64-32-images
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251130_115653/exp-fc1-zscore-ep5000-bs2048-lr1e-3-s42-h64-32-images.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251130_115653/exp-fc1-zscore-ep5000-bs2048-lr1e-3-s42-h64-32-images.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:30:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_job.py --datapath data/dataset100K25M_LA.pkl --model_type fc1 --norm zscore --epochs 5000 --batch_size 2048 --lr 1e-3 --val_split 0.2 --seed 42 --train_type images --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251130_115653/exp-fc1-zscore-ep5000-bs2048-lr1e-3-s42-h64-32-images --weight_decay 0.0 --mag_cost 1.0 --fc1_hidden 64 32 --fc1_activation leaky_relu --fc1_dropout 0.0 --fc1_encoder_enabled --fc1_filter_sizes 3 3 3 --fc1_filter_channels 8 16 32 --fc1_final_embedding_size 16 --fc1_final_embedding_channels 25 --fc1_final_activation none --cpu_only
