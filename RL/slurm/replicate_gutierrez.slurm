#!/bin/bash
#SBATCH --job-name=replicate_gut
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # one core per parallel env (--num-envs 16)
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Usage: sbatch replicate_gutierrez.slurm [total_timesteps] [run_name]
TOTAL=${1:-10000000}
RUN_NAME=${2:-replicate_gutierrez}

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

echo "=== Gutierrez non-coronagraphic replication: ${TOTAL} steps, run ${RUN_NAME} ==="
# Non-coronagraphic Strehl experiment (see PLAN_replicate_gutierrez.md):
#   ideal 21-mode Zernike corrector, 1/f^2 aberration at lambda/4 RMS, in-focus +
#   defocused PSF observation + previous command, Strehl reward -(1-SR)^(2/5).
# Checkpoints every 1M steps (resumable); EvalCallback tracks deterministic Strehl.
python src/train_ppo.py \
    --total-timesteps ${TOTAL} \
    --num-envs 16 \
    --num-control-modes 21 \
    --ideal-modal-correction \
    --num-aberration-modes 21 \
    --aberration-spectrum power_law --psd-exponent 2.0 \
    --no-coronagraph --objective strehl \
    --diversity defocus --defocus-rad 1.0 \
    --include-command \
    --pupil-pixels 256 \
    --max-steps 8 \
    --action-scale 5e-8 \
    --rms-min 0.25 --rms-max 0.25 \
    --learning-rate 3e-4 \
    --log-std-init -1 \
    --log-dir logs/${RUN_NAME} \
    --tb-log-name ${RUN_NAME} \
    --eval-freq 50000 \
    --checkpoint-freq 1000000 \
    --seed 0

echo "=== DONE ${RUN_NAME} ==="
