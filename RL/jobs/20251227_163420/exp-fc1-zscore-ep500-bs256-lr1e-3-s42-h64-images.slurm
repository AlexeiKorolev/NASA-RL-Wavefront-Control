#!/bin/bash
#SBATCH --job-name=exp-fc1-zscore-ep500-bs256-lr1e-3-s42-h64-images
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251227_163420/exp-fc1-zscore-ep500-bs256-lr1e-3-s42-h64-images.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251227_163420/exp-fc1-zscore-ep500-bs256-lr1e-3-s42-h64-images.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_job.py --datapath data/dataset100K25M_LA.pkl --model_type fc1 --norm zscore --epochs 500 --batch_size 256 --lr 1e-3 --val_split 0.2 --seed 42 --train_type images --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251227_163420/exp-fc1-zscore-ep500-bs256-lr1e-3-s42-h64-images --weight_decay 0.0 --mag_cost 1.0 --fc1_hidden 64 --fc1_activation leaky_relu --fc1_dropout 0.0 --fc1_encoder_enabled --fc1_filter_sizes 3 3 3 --fc1_filter_channels 8 16 32 --fc1_final_embedding_size 28 --fc1_final_embedding_channels 32 --fc1_final_activation none --cpu_only
