#!/bin/bash
#SBATCH --job-name=po4probe_r
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

# Continue the M2 learned-probe run deeper (it was still descending, not plateaued).
# Loads the prior best checkpoint (nets + optimizers), no random re-warmup: the loaded
# policy refills the buffer on-policy, then policy training resumes. Architecture args
# MUST match the original run for state_dict loading. Writes a fresh run dir so the
# original logs/checkpoints are preserved.
# Usage: sbatch [--dependency=afterany:<jobid>] slurm/po4ncpa_probe_resume.slurm [resume_ckpt]
RESUME=${1:-logs/po4ncpa_probe/po4ncpa_probe_best.pt}

python src/po4ncpa_probe.py \
    --run-name po4ncpa_probe_r2 \
    --resume "${RESUME}" \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 --probe-frac 0.1 \
    --total-episodes 18000 --warmup-episodes 0 \
    --eval-every 250 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 8 --pol-iters 5 --batch 64 \
    --dyn-lr 1e-3 --pol-lr 1e-4 --explore-std 0.1 \
    --seed 1

echo "=== PO4NCPA PROBE RESUME DONE ==="
