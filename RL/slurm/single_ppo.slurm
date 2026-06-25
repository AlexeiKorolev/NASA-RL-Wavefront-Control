#!/bin/bash
#SBATCH --job-name=single_ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # one core per parallel env (--num-envs 8)
#SBATCH --mem=24G
#SBATCH --time=08:00:00
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

# Quick check of the single-step flat-obs path before the long run (aborts on failure).
python - <<'PY'
from stable_baselines3.common.env_checker import check_env
from environment import CoronagraphEnv
env = CoronagraphEnv(num_control_modes=21, ideal_modal_correction=True,
                     num_aberration_modes=21, aberration_spectrum="power_law",
                     rms_min_waves=0.25, rms_max_waves=0.25,
                     use_coronagraph=False, objective="strehl", diversity="defocus",
                     flatten_obs=True, image_scale="linear", max_steps=1, action_scale=2.5e-7)
print("single-step obs space:", env.observation_space, " action:", env.action_space.shape)
check_env(env, warn=True)
print("SINGLE-STEP ENV CHECK PASSED")
PY

# Pure PPO, single-step (max_steps=1): a one-shot contextual-bandit correction.
# action_scale=2.5e-7 lets a single action cover the full correction; no command
# channel (always zero at the one step); larger n_steps for many one-shot samples/update.
python src/train_ppo.py \
    --total-timesteps 2000000 \
    --num-envs 8 --n-steps 128 \
    --num-control-modes 21 --ideal-modal-correction \
    --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
    --no-coronagraph --objective strehl --diversity defocus --defocus-rad 1.0 \
    --flatten-obs --image-scale linear --net-arch 256,256 \
    --pupil-pixels 256 --max-steps 1 --action-scale 2.5e-7 \
    --rms-min 0.25 --rms-max 0.25 \
    --learning-rate 3e-4 --ent-coef 0.001 --log-std-init -1 \
    --log-dir logs/single_ppo --tb-log-name single_ppo \
    --eval-freq 25000 --checkpoint-freq 500000 --seed 0

echo "=== SINGLE-STEP PPO DONE ==="
