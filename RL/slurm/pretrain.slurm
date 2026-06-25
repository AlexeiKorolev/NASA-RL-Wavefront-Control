#!/bin/bash
#SBATCH --job-name=bc_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # one core per data-generation worker
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2
cd /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL
export PYTHONPATH=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/src:$PYTHONPATH

# One thread per process: the data-generation pool scales with --num-workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Behavior-clone the PPO policy from the least-squares phase-conjugation expert.
# Prints expert sanity check + pre/post-BC deterministic eval, then saves the model.
python src/pretrain_bc.py \
    --num-control-modes 20 \
    --num-actuators-across 16 \
    --pupil-pixels 256 \
    --max-steps 20 \
    --action-scale 1e-8 \
    --rms-min 0.02 --rms-max 0.08 \
    --num-episodes 800 \
    --num-workers 8 \
    --bc-epochs 30 \
    --bc-batch 256 \
    --bc-lr 1e-3 \
    --log-std-init -1 \
    --eval-episodes 30 \
    --out logs/bc_modal20/bc_policy
