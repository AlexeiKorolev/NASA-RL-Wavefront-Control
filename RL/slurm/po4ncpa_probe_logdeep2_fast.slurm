#!/bin/bash
#SBATCH --job-name=po4logd2f
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

# logdeep2, SPED UP (safe pack -- no change to learning dynamics):
#   * --gres=gpu:1 UNPINNED: with every A100 (h11g1, full till ~07-05) and MIG slice
#     (h11g2, 8/8) currently occupied and the V100 node down, take whichever GPU frees
#     first. A MIG slice can backfill sooner than an A100 (the queued 2-A100 job can't use
#     it), so unpinning minimizes queue time. batch=256 fits a 20 GB MIG slice fine.
#   * async overlap of collection with the GPU update phase (train() pipeline): reclaims
#     the ~84% of wall the 14 CPU workers previously sat idle during updates.
#   * batch 64->256 with dyn-iters 8->2 / pol-iters 5->2: SAME samples/round (2x256~=8x64)
#     but 4x bigger, 4x fewer kernels -> feeds the GPU properly (was 14% mem at batch 64).
# Everything else is identical to po4ncpa_probe_logdeep2.slurm so the campaign stays comparable.
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
    --ensemble 5 --ch 64 --dyn-iters 2 --pol-iters 2 --batch 256 \
    --dyn-lr 1e-3 --pol-lr 5e-5 \
    --seed 4

echo "=== PO4NCPA PROBE LOGDEEP2 (fast: A100 + async overlap + big batch) DONE ==="
