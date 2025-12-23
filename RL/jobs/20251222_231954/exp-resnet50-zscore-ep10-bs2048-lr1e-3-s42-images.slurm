#!/bin/bash
#SBATCH --job-name=exp-resnet50-zscore-ep10-bs2048-lr1e-3-s42-images
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251222_231954/exp-resnet50-zscore-ep10-bs2048-lr1e-3-s42-images.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251222_231954/exp-resnet50-zscore-ep10-bs2048-lr1e-3-s42-images.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_job.py --datapath data/dataset100K25M_orth.pkl --model_type resnet50 --norm zscore --epochs 10 --batch_size 2048 --lr 1e-3 --val_split 0.2 --seed 42 --train_type images --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251222_231954/exp-resnet50-zscore-ep10-bs2048-lr1e-3-s42-images --weight_decay 0.0 --mag_cost 1.0 --cpu_only
