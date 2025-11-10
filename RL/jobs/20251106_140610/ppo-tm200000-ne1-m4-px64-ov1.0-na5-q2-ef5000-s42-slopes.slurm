#!/bin/bash
#SBATCH --job-name=ppo-tm200000-ne1-m4-px64-ov1.0-na5-q2-ef5000-s42-slopes
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251106_140610/ppo-tm200000-ne1-m4-px64-ov1.0-na5-q2-ef5000-s42-slopes.out
#SBATCH --error=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251106_140610/ppo-tm200000-ne1-m4-px64-ov1.0-na5-q2-ef5000-s42-slopes.err
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G            # memory per cpu-core (4G is default)
#SBATCH --time=08:00:00
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2

python train_ppo.py --total-timesteps 200000 --num-envs 1 --num-modes 4 --basis zernike --pixels 64 --oversizing-factor 1.0 --num-airy 5 --ppsr 2 --log-dir /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/logs/20251106_140610/ppo-tm200000-ne1-m4-px64-ov1.0-na5-q2-ef5000-s42-slopes --tb-log-name ppo-tm200000-ne1-m4-px64-ov1.0-na5-q2-ef5000-s42-slopes --eval-freq 5000 --seed 42 --include-slopes
