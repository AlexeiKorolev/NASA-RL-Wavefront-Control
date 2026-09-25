#!/bin/bash
#SBATCH --job-name=po4dyn
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

# M4: learned probe vs DYNAMIC aberrations (AR(1) frozen-flow-like drift, constant RMS,
# correlation time tau in control steps -- see PLAN_active_probing.md). The disturbance
# takes one drift step before every control frame, so each action acts on a field one
# step staler than its observation; a policy that exploits temporal correlation
# (predict-ahead) can beat static EFC, which re-chases each frame
# (baseline: slurm/efc_probe_dynamic.slurm). Warm-started from logdeep3 (strong static
# null-digging prior) with a FRESH buffer (static transitions are off-distribution here);
# moderate re-adaptation schedule, same recipe as the mismatch experiment.
# Usage: sbatch slurm/po4ncpa_probe_dynamic.slurm [tau] [episodes]
TAU=${1:-10}
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
echo "=== learned probe vs dynamic NCPA: tau=${TAU}, episodes=${EPISODES} ==="

python src/po4ncpa_infogain.py \
    --run-name po4ncpa_dynamic_t${TAU} \
    --resume logs/po4ncpa_probe_logdeep3/po4ncpa_infogain_best.pt \
    --resume-buffer logs/_force_fresh_buffer_does_not_exist.npz \
    --aberration-tau ${TAU} \
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

echo "=== PO4NCPA DYNAMIC (tau=${TAU}) DONE ==="
