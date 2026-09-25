#!/bin/bash
#SBATCH --job-name=po4mism
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

# PURE RL on the model-MISMATCHED truth (efc_mismatch.py companion).
# The corrector realizes gain_i*coeff_i (per-mode gain error, sigma=$1); classical EFC,
# built on a nominal-gain model, craters here (sigma=0.20 -> probe-EFC 2.1e-7). The RL
# agent trains AND evals on the same gained truth, so it learns the true system and should
# hold ~3.7e-8 -- a clean RL-beats-EFC-under-mismatch result. Warm-started from logdeep3
# (a strong null-digging prior) but with a FRESH buffer (--resume-buffer points nowhere so
# logdeep3's gain=1 transitions are NOT loaded); moderate re-adaptation schedule.
# Usage: sbatch slurm/po4ncpa_probe_mismatch.slurm [sigma] [episodes]
SIGMA=${1:-0.20}
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
echo "=== pure RL on gained truth: sigma=${SIGMA}, episodes=${EPISODES} ==="

python src/po4ncpa_infogain.py \
    --run-name po4ncpa_mismatch_s${SIGMA} \
    --resume logs/po4ncpa_probe_logdeep3/po4ncpa_infogain_best.pt \
    --resume-buffer logs/_force_fresh_buffer_does_not_exist.npz \
    --correction-gain-sigma ${SIGMA} --correction-gain-seed 12345 \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 --probe-frac 0.1 \
    --num-workers 14 --max-steps 8 --lambda-info 0.0 \
    --reward log --lr-final-frac 0.3 --explore-std 0.06 --explore-std-final 0.02 \
    --dyn-lr 3e-4 --pol-lr 2e-5 \
    --resume-refill 20000 --buffer-save-every 10000 \
    --total-episodes ${EPISODES} --warmup-episodes 0 \
    --eval-every 250 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 2 --pol-iters 2 --batch 256 \
    --seed 7

echo "=== PO4NCPA MISMATCH (sigma=${SIGMA}) DONE ==="
