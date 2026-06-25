#!/bin/bash
#SBATCH --job-name=replicate_gut_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # one core per parallel env (--num-envs 16)
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Usage: sbatch replicate_gutierrez_v2.slurm [total_timesteps] [run_name]
TOTAL=${1:-10000000}
RUN_NAME=${2:-replicate_gutierrez_v2}

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

# Abort with a non-zero exit on any failure so a mid-run crash emails FAIL, not END.
set -eo pipefail

echo "=== Gutierrez replication v2: ${TOTAL} steps, run ${RUN_NAME} ==="
# v2 fixes over the failed run (flat eval at Strehl ~0.33): flattened image+command
# -> MlpPolicy (fixes the dead CNN critic), linear image scaling (exposes PSF
# structure), and entropy bonus ent-coef=3e-3 (prevents the std->0.015 collapse).
python src/train_ppo.py \
    --total-timesteps ${TOTAL} \
    --num-envs 16 \
    --num-control-modes 21 --ideal-modal-correction \
    --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
    --no-coronagraph --objective strehl --diversity defocus --defocus-rad 1.0 \
    --include-command --flatten-obs --image-scale linear --net-arch 256,256 \
    --pupil-pixels 256 --max-steps 8 --action-scale 5e-8 \
    --rms-min 0.25 --rms-max 0.25 \
    --learning-rate 3e-4 --ent-coef 3e-3 --log-std-init -1 \
    --log-dir logs/${RUN_NAME} --tb-log-name ${RUN_NAME} \
    --eval-freq 50000 --checkpoint-freq 1000000 --seed 0

echo "=== DONE ${RUN_NAME} ==="
