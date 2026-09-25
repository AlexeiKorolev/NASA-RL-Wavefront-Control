#!/bin/bash
#SBATCH --job-name=po4smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# SMOKE TEST for the mismatch/hybrid code paths (gain in workers + EFC-in-reset + fresh-buffer
# resume). Short run that forces updates to start early (--resume-refill 800) so it exercises
# collection, EFC warm-start, dynamics/policy updates, eval, and checkpoint save without a bug
# lurking for hours. NOT a science run -- 800 episodes only.
# Usage: sbatch slurm/po4ncpa_mismatch_smoke.slurm [efc_warmstart true|false] [sigma]
WARMSTART=${1:-false}
SIGMA=${2:-0.20}

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
echo "=== SMOKE: efc_warmstart=${WARMSTART} sigma=${SIGMA} ==="

python src/po4ncpa_infogain.py \
    --run-name po4ncpa_smoke_ws${WARMSTART}_s${SIGMA} \
    --resume logs/po4ncpa_probe_logdeep3/po4ncpa_infogain_best.pt \
    --resume-buffer logs/_force_fresh_buffer_does_not_exist.npz \
    --correction-gain-sigma ${SIGMA} --correction-gain-seed 12345 \
    --efc-warmstart ${WARMSTART} --efc-warmstart-iters 6 \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 --probe-frac 0.1 \
    --num-workers 8 --max-steps 8 --lambda-info 0.0 \
    --reward log --lr-final-frac 1.0 --explore-std 0.06 --explore-std-final 0.06 \
    --dyn-lr 3e-4 --pol-lr 2e-5 \
    --resume-refill 800 --buffer-save-every 100000 \
    --total-episodes 800 --warmup-episodes 0 \
    --eval-every 200 --eval-episodes 15 \
    --ensemble 5 --ch 64 --dyn-iters 2 --pol-iters 2 --batch 256 \
    --seed 7

echo "=== SMOKE (ws=${WARMSTART} sigma=${SIGMA}) DONE ==="
