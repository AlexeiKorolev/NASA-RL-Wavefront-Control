#!/bin/bash
#SBATCH --job-name=corona_lowab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # one core per parallel env (--num-envs 16)
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Usage: sbatch corona_lowab.slurm <rms_waves> <run_name>
#   <rms_waves>  fixed aberration RMS in waves, e.g. 0.02 / 0.05 / 0.10
#   <run_name>   log dir suffix, e.g. rms02 / rms05 / rms10
RMS=${1:?rms (waves) required}
RUN=${2:?run name required}

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

# Coronagraphic dark-hole control in the SPACE-TELESCOPE regime: small residual
# aberration (start Strehl >> 0.9), where contrast is a locally well-behaved objective
# and the PSF-wrecking degeneracy of the non-coronagraphic case cannot occur (the
# vortex+Lyot removes the bright core there is nothing to smear). Same working recipe
# as replicate_corona.slurm (gamma=0, 4-step, absolute action, 21 ideal Zernike modes,
# 1/f^2, log-scaled coronagraphic sensing, reward = -log10(contrast)/10), but with a
# fixed low RMS swept across runs to map the achievable contrast vs. starting error.
python src/train_ppo.py \
    --total-timesteps 10000000 \
    --num-envs 16 --n-steps 500 --batch-size 500 \
    --num-control-modes 21 --ideal-modal-correction --action-mode absolute \
    --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
    --objective log_contrast --dark-hole-iwa 2 --dark-hole-owa 4 \
    --diversity defocus --defocus-rad 1.8138 \
    --include-command --flatten-obs --image-scale log --log-floor -10 --log-ceil -2 \
    --q 2 --num-airy 4 \
    --pupil-pixels 256 --max-steps 4 --action-scale 3e-7 \
    --rms-min $RMS --rms-max $RMS \
    --learning-rate 1e-3 --lr-linear-decay --gamma 0 --ent-coef 0.0 \
    --target-kl 0 --log-std-init 0.0 \
    --log-dir logs/corona_lowab_$RUN --tb-log-name corona_lowab_$RUN \
    --eval-freq 50000 --checkpoint-freq 1000000 --seed 0

echo "=== CORONA LOW-AB (rms=$RMS -> $RUN) DONE ==="
