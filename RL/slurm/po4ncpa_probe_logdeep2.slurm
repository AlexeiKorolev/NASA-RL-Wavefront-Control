#!/bin/bash
#SBATCH --job-name=po4logdeep
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

# Stable deep-dig against the ~1.4e-6 fixed point, resuming from the M2 best. Combines all
# four fixes from the instability diagnosis:
#   reward=log        -> constant gradient per decade near the floor (breaks the flat-bottom
#                        stall that pins the raw -||o'||^2 policy)
#   lr_final_frac 0.1 -> linear LR decay so it SETTLES instead of orbiting the fixed point
#   explore decay     -> 0.10 -> 0.02, cleaner gradients as it converges
#   resume_refill     -> hold the loaded model/policy until the buffer refills (no transient)
# Compare against the raw-reward, constant-LR run (po4ncpa_probe_deep) which oscillated.
python src/po4ncpa_infogain.py \
    --run-name po4ncpa_probe_logdeep2 \
    --resume logs/po4ncpa_probe_logdeep/po4ncpa_infogain_best.pt \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 --probe-frac 0.1 \
    --num-workers 14 --max-steps 8 --lambda-info 0.0 \
    --reward log --lr-final-frac 0.1 --explore-std 0.1 --explore-std-final 0.02 \
    --resume-refill 20000 \
    --total-episodes 40000 --warmup-episodes 0 \
    --eval-every 250 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 8 --pol-iters 5 --batch 64 \
    --dyn-lr 1e-3 --pol-lr 5e-5 \
    --seed 4

echo "=== PO4NCPA PROBE LOGDEEP (stable, log reward) DONE ==="
