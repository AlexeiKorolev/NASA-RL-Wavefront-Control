#!/bin/bash
#SBATCH --job-name=paper_sanity
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
set -eo pipefail

# Verify the exact Gutierrez config: 16x16 images, 533-dim input, 4-step gamma=0
# episodes, absolute action, expert reachable within the action range.
python - <<'PY'
import numpy as np
from stable_baselines3.common.env_checker import check_env
from environment import CoronagraphEnv
from expert import LeastSquaresExpert

A = 3e-7   # action_scale
kw = dict(num_control_modes=21, ideal_modal_correction=True, action_mode="absolute",
          num_aberration_modes=21, aberration_spectrum="power_law", psd_exponent=2.0,
          rms_min_waves=0.25, rms_max_waves=0.25,
          q=2, num_airy=4.0,
          use_coronagraph=False, objective="strehl", diversity="defocus", defocus_rad=1.8138,
          include_command=True, flatten_obs=True, image_scale="linear", linear_ceil=1.0,
          max_steps=4, action_scale=A)
env = CoronagraphEnv(**kw)
print("image_shape:", env.optics.image_shape, " obs space:", env.observation_space, " action:", env.action_space.shape)
assert env.optics.image_shape == (16, 16), "expected 16x16 images"
check_env(env, warn=True)
print("PAPER ENV CHECK PASSED")

# Expert one-shot Strehl on the 16x16 grid, and reachability of expert coeffs at |a|<=1.
expert = LeastSquaresExpert(env.optics, env.modal_basis)
ss, reach = [], []
for e in range(10):
    _, info = env.reset(seed=9000 + e)
    c = expert.correction()
    reach.append(float(np.max(np.abs(c)) / A))   # <1 => reachable by a clipped action
    env.optics.set_correction_modes(c)
    ss.append(float(np.max(env.optics.normalized_intensity(coronagraph=False, lyot=False))))
print(f"expert one-shot Strehl median {np.median(ss):.4f}  (target ~1.0)")
print(f"max|expert coeff|/action_scale median {np.median(reach):.2f}, worst {np.max(reach):.2f}  (<1 = reachable)")
PY

# Short PPO smoke with the exact paper hyperparameters (gamma 0, lr 1e-3 decay, etc.).
python src/train_ppo.py \
    --total-timesteps 80000 \
    --num-envs 8 --n-steps 500 --batch-size 500 \
    --num-control-modes 21 --ideal-modal-correction --action-mode absolute \
    --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
    --no-coronagraph --objective strehl --diversity defocus --defocus-rad 1.8138 \
    --include-command --flatten-obs --image-scale linear \
    --q 2 --num-airy 4 \
    --pupil-pixels 256 --max-steps 4 --action-scale 3e-7 \
    --rms-min 0.25 --rms-max 0.25 \
    --learning-rate 1e-3 --lr-linear-decay --gamma 0 --ent-coef 0.0 \
    --target-kl 0 --log-std-init 0.0 \
    --log-dir logs/paper_sanity --tb-log-name paper_sanity \
    --eval-freq 20000

echo "PAPER SANITY DONE"
