#!/bin/bash
#SBATCH --job-name=po4prof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
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

# PROFILING ONLY (~10 min). After a full A100 + async overlap + batch 256 gave ZERO wall-time
# speedup, this measures the real per-episode breakdown in the actual train loop:
#   upd (smpl / dcomp)  -- is the update phase dominated by buf.sample host->device copies
#                          (CPU plumbing) or by GPU compute?
#   gath                -- is the async collect/update overlap actually working (~0 = yes)?
#   eval                -- how much do the serial eval episodes cost per window?
# Fresh (not resumed), short, frequent evals. Same env/model config as logdeep2 so numbers
# transfer. NOT a science run -- logs/po4ncpa_probe_profile is throwaway.
python src/po4ncpa_infogain.py \
    --run-name po4ncpa_probe_profile \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 --probe-frac 0.1 \
    --num-workers 14 --max-steps 8 --lambda-info 0.0 --reward log \
    --total-episodes 1000 --warmup-episodes 200 \
    --eval-every 100 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 2 --pol-iters 2 --batch 256 \
    --dyn-lr 1e-3 --pol-lr 5e-5 \
    --seed 7

echo "=== PO4NCPA PROFILE DONE ==="
