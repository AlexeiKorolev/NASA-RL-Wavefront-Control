#!/bin/bash
#SBATCH --job-name=po4dynscr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# M4, take 2: dynamic NCPA trained from SCRATCH (no logdeep3 warm-start), at a FIXED tau.
# The warm-started run (logs/po4ncpa_dynamic_t10, job 3313383) stalled flat at ~3.5e-5 for
# 15k+ episodes -- no better than the static-trained policy just sitting under drift,
# unchanged from the smoke test. Hypothesis: the dynamics net was fit on zero-drift data and
# its (obs, prev_action, action) -> next-image mapping is deterministic by construction; the
# warm start anchored it near that static-world fit instead of letting it re-learn the
# drift-injected residual energy from scratch. This run drops --resume entirely so the
# dynamics ensemble and policy fit the drifting truth from the start, same recipe as the
# original static from-scratch baseline (po4ncpa_probe_fast.slurm) but with reward=log
# from the outset (raw reward is known to stall in a flat-gradient bottom near the floor --
# no reason to repeat that mistake here) and proper warmup restored (irrelevant under
# --resume, but essential from scratch: warmup episodes use uniform-random actions so the
# dynamics net sees a diverse buffer before policy updates start).
# Baseline to beat: EFC's tau=10 tracking floor (mean of last 3/20 iterations, drifting
# every iteration) = 4.15e-8 (slurm/efc_probe_dynamic.slurm, job 3313341).
# Usage: sbatch slurm/po4ncpa_probe_dynamic_scratch.slurm [tau] [episodes]
TAU=${1:-10}
EPISODES=${2:-40000}

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
set -eo pipefail

nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "=== dynamic NCPA FROM SCRATCH: tau=${TAU}, episodes=${EPISODES} ==="

python src/po4ncpa_infogain.py \
    --run-name po4ncpa_dynamic_scratch_t${TAU} \
    --aberration-tau ${TAU} \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 --probe-frac 0.1 \
    --num-workers 14 --max-steps 8 --lambda-info 0.0 \
    --reward log --lr-final-frac 0.3 --explore-std 0.1 --explore-std-final 0.02 \
    --total-episodes ${EPISODES} --warmup-episodes 4000 \
    --eval-every 250 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 8 --pol-iters 5 --batch 64 \
    --dyn-lr 1e-3 --pol-lr 1e-4 \
    --seed 11

echo "=== PO4NCPA DYNAMIC SCRATCH (tau=${TAU}) DONE ==="
