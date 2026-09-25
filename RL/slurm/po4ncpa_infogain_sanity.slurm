#!/bin/bash
#SBATCH --job-name=po4ig_san
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
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

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
set -eo pipefail

# M2-honest smoke test: shapes / parallel-collection plumbing / runtime. Watch that the
# 'info' column (imagined info-gain reward for the probe) is POSITIVE and trending up --
# that is the new observable-only sensing signal doing its job.
python src/po4ncpa_infogain.py \
    --run-name po4ncpa_infogain_sanity \
    --total-episodes 800 --warmup-episodes 200 \
    --eval-every 100 --eval-episodes 10 \
    --num-workers 14 --max-steps 8 \
    --ensemble 3 --ch 32 --lambda-info 0.3 \
    --seed 0

echo "=== PO4NCPA INFOGAIN SANITY DONE ==="
