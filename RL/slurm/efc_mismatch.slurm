#!/bin/bash
#SBATCH --job-name=efc_mismatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# EFC under corrector-gain model mismatch (notes/PLAN_active_probing.md).
# Builds EFC's model at nominal gain, runs the loop on a gain-mismatched truth, sweeps the
# mismatch magnitude. This is the cheap, no-training half of the "why not EFC" experiment:
# if EFC does NOT degrade here, the whole RL premise is moot; if it does, the learned probe
# (3.70e-8 on the true system) wins past some mismatch level.
# Usage: sbatch slurm/efc_mismatch.slurm [n_draws] [efc_iters] [gain_sigma_list]
NDRAWS=${1:-15}
ITERS=${2:-10}
SIGMAS=${3:-0,0.02,0.05,0.1,0.2,0.3}

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

python src/efc_mismatch.py \
    --checkpoint logs/po4ncpa_corona/po4ncpa_best.pt \
    --n-draws "${NDRAWS}" --efc-iters "${ITERS}" \
    --gain-sigma-list "${SIGMAS}" \
    --out-csv logs/efc_mismatch/efc_mismatch.csv

echo "=== EFC MISMATCH SWEEP DONE ==="
