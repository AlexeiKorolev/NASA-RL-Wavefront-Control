#!/bin/bash
#SBATCH --job-name=exp-fc1-minmax-ep10000-bs2048-lr1e-3-s42-h25-images
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251128_153811/exp-fc1-minmax-ep10000-bs2048-lr1e-3-s42-h25-images.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251128_153811/exp-fc1-minmax-ep10000-bs2048-lr1e-3-s42-h25-images.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=03:30:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_job.py --datapath data/dataset100K25M_VLA.pkl --model_type fc1 --norm minmax --epochs 10000 --batch_size 2048 --lr 1e-3 --val_split 0.2 --seed 42 --train_type images --out_dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251128_153811/exp-fc1-minmax-ep10000-bs2048-lr1e-3-s42-h25-images --fc1_hidden 25 --fc1_activation none --fc1_final_activation none --cpu_only
