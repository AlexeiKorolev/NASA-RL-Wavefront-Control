#!/bin/bash
#SBATCH --job-name=po4_strehl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2
cd /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL
export PYTHONPATH=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/src:$PYTHONPATH

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
set -eo pipefail

# PO4NCPA, non-coronagraphic Strehl task (paper Sec. static NCPA, SI), PAPER-FAITHFUL
# parameters: 55 Zernike modes from Noll 4 (piston+tip/tilt removed), Kolmogorov-like
# modal spectrum (1/f^(11/3)), 286nm-equivalent RMS (0.026 waves), 4000-episode
# random warm-up and ~30k total episodes (paper converges by ~50k). Target: match the
# paper's static-SI Strehl ~0.994.
python src/po4ncpa.py \
    --run-name po4ncpa_strehl \
    --use-coronagraph false \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 \
    --total-episodes 30000 --warmup-episodes 4000 \
    --eval-every 250 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 8 --pol-iters 5 --batch 64 \
    --dyn-lr 1e-3 --pol-lr 1e-4 --explore-std 0.1 \
    --seed 0

echo "=== PO4NCPA STREHL DONE ==="
