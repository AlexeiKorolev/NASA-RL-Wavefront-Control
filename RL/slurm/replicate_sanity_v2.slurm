#!/bin/bash
#SBATCH --job-name=replicate_sanity_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
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

# Abort (non-zero exit) on any failure so the dependent long run's afterok gate
# actually triggers and the completion email reports FAIL rather than END.
set -eo pipefail

# Phase-0 sanity for the v2 fixes: flattened image+command -> MlpPolicy, linear
# image scaling, entropy bonus. Gates the long run via a SLURM afterok dependency.
python - <<'PY'
import numpy as np
from stable_baselines3.common.env_checker import check_env
from environment import CoronagraphEnv

kw = dict(num_control_modes=21, ideal_modal_correction=True,
          num_aberration_modes=21, aberration_spectrum="power_law", psd_exponent=2.0,
          rms_min_waves=0.25, rms_max_waves=0.25,
          use_coronagraph=False, objective="strehl", diversity="defocus", defocus_rad=1.0,
          include_command=True, flatten_obs=True, image_scale="linear",
          max_steps=8, action_scale=5e-8)

env = CoronagraphEnv(**kw)
print("obs space:", env.observation_space, " expected dim:", 2*56*56 + 21)
print("action dim:", env.action_space.shape)
check_env(env, warn=True)
obs, info = env.reset(seed=4242)
print(f"flat obs shape {obs.shape}, range [{obs.min():.3f}, {obs.max():.3f}], start Strehl {info['strehl']:.4f}")
print("V2 ENV CHECK PASSED")
PY

# Short PPO smoke on the full v2 config (MlpPolicy via --flatten-obs).
python src/train_ppo.py \
    --total-timesteps 12000 \
    --num-envs 2 \
    --num-control-modes 21 --ideal-modal-correction \
    --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
    --no-coronagraph --objective strehl --diversity defocus --defocus-rad 1.0 \
    --include-command --flatten-obs --image-scale linear --net-arch 256,256 \
    --pupil-pixels 256 --max-steps 8 --action-scale 5e-8 \
    --rms-min 0.25 --rms-max 0.25 \
    --learning-rate 3e-4 --ent-coef 3e-3 --log-std-init -1 \
    --log-dir logs/replicate_sanity_v2 --tb-log-name replicate_sanity_v2 \
    --eval-freq 4000

echo "REPLICATE SANITY V2 DONE"
