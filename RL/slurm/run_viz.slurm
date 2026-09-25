#!/bin/bash
#SBATCH --job-name=po4_viz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Render the PO4NCPA visualization figures (plain python, no Jupyter kernel needed:
# the dmrl2 env has matplotlib+torch but not ipykernel/nbconvert).
# Usage: sbatch slurm/run_viz.slurm [run_name] [tag]   e.g. sbatch slurm/run_viz.slurm po4ncpa_corona best
# Produces: notebooks/figs/view{1,2,3,4}_*.png

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

RUN=${1:-po4ncpa_corona}
TAG=${2:-best}

python src/viz_po4ncpa.py \
    --checkpoint logs/${RUN}/po4ncpa_${TAG}.pt \
    --out notebooks/figs \
    --demo-seed 900000 \
    --episodes 60

echo "=== VIZ DONE -> notebooks/figs/view{1,2,3,4}_*.png (${RUN}/${TAG}) ==="
