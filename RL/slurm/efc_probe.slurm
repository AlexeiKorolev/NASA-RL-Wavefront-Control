#!/bin/bash
#SBATCH --job-name=efc_probe_m1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# M1 pairwise-probe EFC baseline (notes/PLAN_active_probing.md)
# Usage: sbatch slurm/efc_probe.slurm [checkpoint] [n_draws] [efc_iters]
CKPT=${1:-logs/po4ncpa_corona/po4ncpa_best.pt}
NDRAWS=${2:-15}
ITERS=${3:-10}

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

python src/efc_probe.py --checkpoint "${CKPT}" --n-draws "${NDRAWS}" --efc-iters "${ITERS}"

echo "=== EFC PROBE M1 DONE ==="
