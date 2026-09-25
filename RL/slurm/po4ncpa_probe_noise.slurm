#!/bin/bash
#SBATCH --job-name=po4noise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Usage: sbatch slurm/po4ncpa_probe_noise.slurm <FLUX>
# M3: learned reward-only probe (lambda_info=0) trained UNDER photon noise. Each exposure
# (dark image + both probe images) is independently shot-noise-limited at FLUX photons/peak,
# so the policy must learn a probe/correction robust to sensing noise. Eval reports TRUE
# noiseless contrast, directly comparable to the EFC-vs-flux curve from efc_probe_noise.
FLUX=${1:?pass photon flux as arg1 (e.g. 1e9)}

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

python src/po4ncpa_infogain.py \
    --run-name po4ncpa_probe_noise_f${FLUX} \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 --probe-frac 0.1 \
    --photon-flux ${FLUX} \
    --num-workers 14 --max-steps 8 --lambda-info 0.0 \
    --total-episodes 25000 --warmup-episodes 2000 \
    --eval-every 250 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 8 --pol-iters 5 --batch 64 \
    --dyn-lr 1e-3 --pol-lr 1e-4 --explore-std 0.1 \
    --seed 0

echo "=== PO4NCPA PROBE NOISE flux=${FLUX} DONE ==="
