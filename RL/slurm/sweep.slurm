#!/bin/bash
#SBATCH --job-name=gut_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # one core per parallel env (--num-envs 8)
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Usage: sbatch sweep.slurm <image_scale> <ent_coef> <action_scale> <run_name>
IMAGE_SCALE=$1
ENT_COEF=$2
ACTION_SCALE=$3
RUN_NAME=$4

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

echo "=== sweep ${RUN_NAME}: image_scale=${IMAGE_SCALE} ent_coef=${ENT_COEF} action_scale=${ACTION_SCALE} ==="
# Short (400k-step) diagnostic on the Gutierrez non-corono config, MLP architecture.
# Isolates which of the three v2 knobs drives learning vs. the Strehl<0.01 divergence.
python src/train_ppo.py \
    --total-timesteps 400000 \
    --num-envs 8 \
    --num-control-modes 21 --ideal-modal-correction \
    --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
    --no-coronagraph --objective strehl --diversity defocus --defocus-rad 1.0 \
    --include-command --flatten-obs --image-scale ${IMAGE_SCALE} --net-arch 256,256 \
    --pupil-pixels 256 --max-steps 8 --action-scale ${ACTION_SCALE} \
    --rms-min 0.25 --rms-max 0.25 \
    --learning-rate 3e-4 --ent-coef ${ENT_COEF} --log-std-init -1 \
    --log-dir logs/${RUN_NAME} --tb-log-name ${RUN_NAME} \
    --eval-freq 25000 --seed 0

echo "=== DONE ${RUN_NAME} ==="
