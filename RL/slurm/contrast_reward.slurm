#!/bin/bash
#SBATCH --job-name=creward
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # one core per parallel env (--num-envs 16)
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Usage: sbatch contrast_reward.slurm <objective> <run_name> [linear_ceil]
#   <objective>   log_contrast | neg_contrast | contrast
#   <run_name>    log dir suffix, e.g. logc / negc / diffc
#   [linear_ceil] only used by neg_contrast (default 0.01)
OBJ=${1:?objective required}
RUN=${2:?run name required}
LCEIL=${3:-0.01}

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

# Non-coronagraphic wavefront control with the proven paper recipe (gamma=0, 4-step,
# absolute action, 21 ideal Zernike modes, 1/f^2 lambda/4, 16x16 + command -> MLP),
# but rewarded by the BARE-PSF dark-hole annulus contrast (2-4 lambda/D) instead of
# Strehl. Objective selects the reward SHAPE:
#   log_contrast : r = -log10(C)/10     (absolute, log)
#   neg_contrast : r = -C/linear_ceil   (absolute, linear)
#   contrast     : r = log10(C_prev/C)  (differential, log)
python src/train_ppo.py \
    --total-timesteps 10000000 \
    --num-envs 16 --n-steps 500 --batch-size 500 \
    --num-control-modes 21 --ideal-modal-correction --action-mode absolute \
    --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
    --no-coronagraph --objective $OBJ --dark-hole-iwa 2 --dark-hole-owa 4 \
    --diversity defocus --defocus-rad 1.8138 \
    --include-command --flatten-obs --image-scale linear --linear-ceil $LCEIL \
    --q 2 --num-airy 4 \
    --pupil-pixels 256 --max-steps 4 --action-scale 3e-7 \
    --rms-min 0.25 --rms-max 0.25 \
    --learning-rate 1e-3 --lr-linear-decay --gamma 0 --ent-coef 0.0 \
    --target-kl 0 --log-std-init 0.0 \
    --log-dir logs/creward_$RUN --tb-log-name creward_$RUN \
    --eval-freq 50000 --checkpoint-freq 1000000 --seed 0

echo "=== CONTRAST REWARD ($OBJ -> $RUN) DONE ==="
