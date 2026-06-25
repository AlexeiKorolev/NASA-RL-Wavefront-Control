#!/bin/bash
#SBATCH --job-name=single_bc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
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

# Single-step behavior cloning: regress the policy onto the expert's one-shot
# correction. The expert hits Strehl ~1.0 in one shot, so BC should yield a
# near-perfect single-step corrector. Reliable counterpart to the pure-PPO run.
python src/pretrain_bc.py \
    --num-control-modes 21 --ideal-modal-correction \
    --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
    --no-coronagraph --objective strehl --diversity defocus --defocus-rad 1.0 \
    --flatten-obs --image-scale linear --net-arch 256,256 \
    --pupil-pixels 256 --max-steps 1 --action-scale 2.5e-7 \
    --rms-min 0.25 --rms-max 0.25 \
    --num-episodes 20000 --num-workers 8 \
    --bc-epochs 50 --bc-batch 256 --bc-lr 1e-3 \
    --eval-episodes 50 \
    --out logs/single_bc/bc_policy

echo "=== SINGLE-STEP BC DONE ==="
