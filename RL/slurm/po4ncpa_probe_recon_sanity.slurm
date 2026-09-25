#!/bin/bash
#SBATCH --job-name=po4recon_san
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
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

# M2-refine smoke test: shapes/runtime for the reconstruction-auxiliary trainer.
# Watch that fldL (field-reconstruction loss) decreases -- that is the new signal.
python src/po4ncpa_probe_recon.py \
    --run-name po4ncpa_probe_recon_sanity \
    --total-episodes 400 --warmup-episodes 200 \
    --eval-every 100 --eval-episodes 10 \
    --ensemble 3 --ch 32 \
    --seed 0

echo "=== PO4NCPA PROBE RECON SANITY DONE ==="
