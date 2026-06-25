#!/bin/bash
#SBATCH --job-name=po4_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Usage: sbatch slurm/eval_po4ncpa.slurm [run_name] [tag] [episodes]
#   e.g. sbatch slurm/eval_po4ncpa.slurm po4ncpa_corona best 150
RUN=${1:-po4ncpa_corona}
TAG=${2:-best}
EP=${3:-150}

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

python src/eval_po4ncpa.py \
    --checkpoint logs/${RUN}/po4ncpa_${TAG}.pt \
    --episodes ${EP} \
    --seed-offset 900000

echo "=== PO4NCPA EVAL DONE (${RUN}/${TAG}) ==="
