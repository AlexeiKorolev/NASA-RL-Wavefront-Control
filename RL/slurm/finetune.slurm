#!/bin/bash
#SBATCH --job-name=ppo_finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # one core per parallel env
#SBATCH --mem=32G
#SBATCH --time=01:45:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2
cd /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL
export PYTHONPATH=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/src:$PYTHONPATH

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# RL fine-tune from the behavior-cloned policy (logs/bc_modal20/bc_policy.zip).
# Low lr + low exploration std protect the warm start; PPO optimizes the actual
# dark-hole contrast reward, which can exceed the phase-conjugation expert.
# EvalCallback saves the best deterministic checkpoint, so a degrading run can't
# lose the warm-started performance.
python src/train_ppo.py \
    --init-model logs/bc_modal20/bc_policy \
    --total-timesteps 100000 \
    --num-envs 8 \
    --num-actuators-across 16 \
    --num-control-modes 20 \
    --pupil-pixels 256 \
    --max-steps 20 \
    --action-scale 1e-8 \
    --rms-min 0.02 --rms-max 0.08 \
    --learning-rate 5e-5 \
    --log-std-init -2.5 \
    --log-dir logs/ppo_finetune_modal20 \
    --tb-log-name ppo_finetune_modal20 \
    --eval-freq 10000
