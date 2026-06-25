#!/bin/bash
#SBATCH --job-name=ideal_ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # one core per parallel env (--num-envs 8)
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Usage: sbatch ideal_zernike.slurm <rms_min> <rms_max> <run_name>
#   e.g. sbatch ideal_zernike.slurm 0.02 0.08 ideal_lownoise
#        sbatch ideal_zernike.slurm 0.10 0.30 ideal_highnoise
RMS_MIN=${1:-0.02}
RMS_MAX=${2:-0.08}
RUN_NAME=${3:-ideal_zernike}

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

echo "=== ideal-corrector PPO: rms ${RMS_MIN}-${RMS_MAX}, run ${RUN_NAME} ==="

# Model-free PPO from scratch on the ideal Zernike corrector (no DM bandwidth
# limit). Same hyperparameters across both noise levels so the learning curves
# (EvalCallback return + env/log10_contrast) are directly comparable.
python src/train_ppo.py \
    --total-timesteps 200000 \
    --num-envs 8 \
    --num-control-modes 20 \
    --ideal-modal-correction \
    --pupil-pixels 256 \
    --max-steps 20 \
    --action-scale 1e-8 \
    --rms-min ${RMS_MIN} --rms-max ${RMS_MAX} \
    --learning-rate 3e-4 \
    --log-std-init -1 \
    --log-dir logs/${RUN_NAME} \
    --tb-log-name ${RUN_NAME} \
    --eval-freq 10000 \
    --seed 0

echo "=== DONE ${RUN_NAME} ==="
