#!/bin/bash
#SBATCH --job-name=po4deep
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

# Deep-resume the plain reward-only probe (M2 best, 1.39e-6) and train it MUCH longer with
# the parallel-collection speedup. Tests whether the "plateau" is just slow convergence: the
# original run was still descending when it stopped. max_steps=20 matches the original (no
# confound); lambda_info=0 = pure control reward; --resume loads policy+dynamics+optimizers.
# 60k more episodes at ~8x speed (~6-8h). Eval reports TRUE noiseless contrast.
python src/po4ncpa_infogain.py \
    --run-name po4ncpa_probe_deep \
    --resume logs/po4ncpa_probe_r2/po4ncpa_probe_best.pt \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 --probe-frac 0.1 \
    --num-workers 14 --max-steps 8 --lambda-info 0.0 \
    --total-episodes 60000 --warmup-episodes 0 \
    --eval-every 250 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 8 --pol-iters 5 --batch 64 \
    --dyn-lr 1e-3 --pol-lr 1e-4 --explore-std 0.1 \
    --seed 2

echo "=== PO4NCPA PROBE DEEP (resume, lambda=0) DONE ==="
