#!/bin/bash
#SBATCH --job-name=po4_sanity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2
cd /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL
export PYTHONPATH=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/src:$PYTHONPATH

# Single process; let the FFT/SVD use a few threads (hcipy propagation is on CPU).
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
set -eo pipefail

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# PO4NCPA smoke test: non-coronagraphic Strehl task with PAPER-FAITHFUL parameters
# (55 Zernike modes from Noll 4 = tip/tilt removed, Kolmogorov-like 1/f^(11/3)
# spectrum, fixed 0.026-wave = 286nm-equivalent RMS). Short run -- verifies the
# optics start-mode change works and that eval Strehl climbs faster/higher than the
# 20-mode / 0.02-0.08-wave config did.
python src/po4ncpa.py \
    --run-name po4ncpa_sanity \
    --use-coronagraph false \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --rms-min 0.026 --rms-max 0.026 \
    --per-mode-scale true --action-headroom 5.0 \
    --total-episodes 2500 --warmup-episodes 600 \
    --eval-every 100 --eval-episodes 10 \
    --ensemble 5 --ch 64 --dyn-iters 8 --pol-iters 5 --batch 64 \
    --seed 0

echo "=== PO4NCPA SANITY DONE ==="
