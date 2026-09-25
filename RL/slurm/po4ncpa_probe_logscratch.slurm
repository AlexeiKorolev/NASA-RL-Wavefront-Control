#!/bin/bash
#SBATCH --job-name=po4logscr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# ---------------------------------------------------------------------------
# LOG REWARD FROM SCRATCH -- static coronagraphic learned-probe PO4NCPA.
#
# The scientific question: the deep static campaign that reached 3.70e-8 used a WEIRD
# piecewise schedule -- raw reward -||o'||^2 first (po4ncpa_probe / _deep, landed ~1.4e-6),
# then a resume chain switched to the log-contrast reward (logdeep/2/3) to escape the
# flat-gradient bottom near the floor. This run isolates the log reward as the SINGLE
# change: train FROM SCRATCH with --reward log from episode 1 (no raw phase, no resume
# chain), proper 4000-ep random warmup, and let it run long enough to settle. Does the
# clean single run dig as deep as the staged chain, or was the piecewise curriculum doing
# real work?
#
# Config is paper-faithful + the fast recipe (14-worker parallel collection, max_steps=8,
# ~18x cheaper): 55 Zernike modes from Noll 4, 0.026-wave RMS, Kolmogorov 1/f^(11/3) modal
# spectrum, per-mode action scaling (headroom 5), probe_frac 0.1, dark hole 1.5-4 lambda/D
# in the 33x33 FoV. lambda_info=0 (the info-gain auxiliary was shown to hurt; pure control
# reward). Eval reports TRUE noiseless contrast (median over 30 held-out draws).
#
# Reference points (same geometry/modes):
#   passive PO4NCPA plateau .......... ~1.5e-5  (sensing wall, no probe)
#   piecewise raw->log campaign best .. 3.70e-8  (logdeep3, ~175k cumulative episodes)
#   ideal 55-mode correction floor .... 1.18e-10
#
# Logging: logs/po4ncpa_probe_logscratch/train.csv (episode, train/eval contrast+strehl,
# info_gain), console -> outputs/slurm-<jobid>.out, best+final checkpoints, and
# replay_buffer.npz every 10k episodes so this run is resumable if it hits the walltime or
# is still descending at 120k (relaunch with --resume logs/.../po4ncpa_infogain_best.pt).
#
# Usage: sbatch slurm/po4ncpa_probe_logscratch.slurm [episodes]
# ---------------------------------------------------------------------------
EPISODES=${1:-120000}

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
echo "=== PO4NCPA LOG-REWARD FROM SCRATCH (static): episodes=${EPISODES} ==="

python src/po4ncpa_infogain.py \
    --run-name po4ncpa_probe_logscratch \
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
    --buffer-save-every 10000 \
    --seed 0

echo "=== PO4NCPA LOG-REWARD FROM SCRATCH DONE ==="
