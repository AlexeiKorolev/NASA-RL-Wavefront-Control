#!/bin/bash
#SBATCH --job-name=exp-tf1-zscore-ep10000-bs262144-lr1e-3-s42-images-ps4-d128-L4-h4-mlp256-cls
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_195013/exp-tf1-zscore-ep10000-bs262144-lr1e-3-s42-images-ps4-d128-L4-h4-mlp256-cls.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_195013/exp-tf1-zscore-ep10000-bs262144-lr1e-3-s42-images-ps4-d128-L4-h4-mlp256-cls.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_job.py --datapath data/dataset220K25M_zernike.pkl --model_type tf1 --norm zscore --epochs 10000 --batch_size 262144 --lr 1e-3 --val_split 0.2 --seed 42 --train_type images --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_195013/exp-tf1-zscore-ep10000-bs262144-lr1e-3-s42-images-ps4-d128-L4-h4-mlp256-cls --tf1_patch_size 4 --tf1_dim 128 --tf1_depth 4 --tf1_heads 4 --tf1_mlp_dim 256 --tf1_attn_dropout 0.0 --tf1_emb_dropout 0.0 --tf1_use_cls_token
