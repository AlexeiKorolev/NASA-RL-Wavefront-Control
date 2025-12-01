#!/bin/bash
#SBATCH --job-name=exp-tf1-zscore-ep4000-bs2048-lr1e-3-s42-images-ps4-d32-L2-h4-mlp64-cls-wd0.001
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251201_091202/exp-tf1-zscore-ep4000-bs2048-lr1e-3-s42-images-ps4-d32-L2-h4-mlp64-cls-wd0.001.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251201_091202/exp-tf1-zscore-ep4000-bs2048-lr1e-3-s42-images-ps4-d32-L2-h4-mlp64-cls-wd0.001.err
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

python train_job.py --datapath data/dataset100K25M_LA.pkl --model_type tf1 --norm zscore --epochs 4000 --batch_size 2048 --lr 1e-3 --val_split 0.2 --seed 42 --train_type images --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251201_091202/exp-tf1-zscore-ep4000-bs2048-lr1e-3-s42-images-ps4-d32-L2-h4-mlp64-cls-wd0.001 --weight_decay 0.001 --mag_cost 1.0 --tf1_patch_size 4 --tf1_dim 32 --tf1_depth 2 --tf1_heads 4 --tf1_mlp_dim 64 --tf1_attn_dropout 0.0 --tf1_emb_dropout 0.0 --tf1_use_cls_token --cpu_only
