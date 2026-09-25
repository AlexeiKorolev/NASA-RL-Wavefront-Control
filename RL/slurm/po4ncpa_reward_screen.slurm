#!/bin/bash
#SBATCH --job-name=po4_reward
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm-%j.out

# Usage: sbatch slurm/po4ncpa_reward_screen.slurm REWARD_SHAPE
# Five 30k-episode screens. All settings match the passive PO4NCPA rerun.
REWARD_SHAPE=$1
case "$REWARD_SHAPE" in
    log_energy|sqrt_energy|masked_log_energy|masked_log_cubic|masked_tail_log_cubic) ;;
    *) echo "Unknown reward shape: $REWARD_SHAPE" >&2; exit 2 ;;
esac

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

python src/po4ncpa.py \
    --run-name "po4ncpa_reward_${REWARD_SHAPE}_20260922" \
    --reward-shape "$REWARD_SHAPE" \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 \
    --total-episodes 30000 --warmup-episodes 4000 \
    --eval-every 250 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 8 --pol-iters 5 --batch 64 \
    --dyn-lr 1e-3 --pol-lr 1e-4 --explore-std 0.1 \
    --seed 0 --save-full-state true
