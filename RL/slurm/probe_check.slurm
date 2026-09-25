#!/bin/bash
#SBATCH --job-name=probe_m0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# M0 active-probing diagnostic: linearization + field reconstruction (notes/PLAN_active_probing.md)
# Usage: sbatch slurm/probe_check.slurm [checkpoint] [n_draws] [n_probes]
CKPT=${1:-logs/po4ncpa_corona/po4ncpa_best.pt}
NDRAWS=${2:-20}
NPROBES=${3:-8}

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

python src/probe_check.py --checkpoint "${CKPT}" --n-draws "${NDRAWS}" --n-probes "${NPROBES}"

echo "=== PROBE M0 DONE ==="
