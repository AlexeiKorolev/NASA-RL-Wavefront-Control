#!/bin/bash
#SBATCH --job-name=po4logd3
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

# Continuation of logdeep2 (which ended at 7.10e-8, STILL descending -- not a plateau; the
# 40k budget was half-eaten recovering from the resume jump). Two changes vs logdeep2:
#   * CONTINUATION SCHEDULE: start where logdeep2 ended so the resume doesn't re-kick the
#     policy -- explore-std 0.02 (not 0.1), pol-lr 5e-6 / dyn-lr 1e-4 (logdeep2's end = its
#     0.1x-decayed LRs). Gentle further decay (lr-final-frac 0.5, explore -> 0.01). This is
#     what actually caused the sawtooth (schedule reset + dynamics re-fit).
#   * Buffer save/reload is now live: logdeep2 has no saved buffer so THIS resume still
#     refills once (mild dip expected), but logdeep3 writes replay_buffer.npz every 10k eps,
#     so any resume from logdeep3 onward skips the refill transient entirely.
# Same env + model config as logdeep2 (incl. batch 256 / dyn 2 / pol 2) for continuity, and
# the optimizer state is loaded from the checkpoint. Goal: dig below 7.10e-8.
python src/po4ncpa_infogain.py \
    --run-name po4ncpa_probe_logdeep3 \
    --resume logs/po4ncpa_probe_logdeep2/po4ncpa_infogain_best.pt \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 --probe-frac 0.1 \
    --num-workers 14 --max-steps 8 --lambda-info 0.0 \
    --reward log --lr-final-frac 0.5 --explore-std 0.02 --explore-std-final 0.01 \
    --resume-refill 20000 --buffer-save-every 10000 \
    --total-episodes 40000 --warmup-episodes 0 \
    --eval-every 250 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 2 --pol-iters 2 --batch 256 \
    --dyn-lr 1e-4 --pol-lr 5e-6 \
    --seed 5

echo "=== PO4NCPA PROBE LOGDEEP3 (continuation, gentle schedule, buffer-saving) DONE ==="
