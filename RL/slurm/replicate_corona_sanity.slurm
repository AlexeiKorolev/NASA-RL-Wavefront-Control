#!/bin/bash
#SBATCH --job-name=corona_sanity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
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

# CORONAGRAPHIC variant of the working recipe: same gamma=0 / 4-step / absolute-action /
# 21-Zernike / lambda-4 setup, but the coronagraph is ON, the agent SENSES from the
# coronagraphic (dark) images (log-scaled), and the reward is dark-hole contrast.
python - <<'PY'
import numpy as np
from stable_baselines3.common.env_checker import check_env
from environment import CoronagraphEnv
from expert import LeastSquaresExpert

A = 3e-7
kw = dict(num_control_modes=21, ideal_modal_correction=True, action_mode="absolute",
          num_aberration_modes=21, aberration_spectrum="power_law", psd_exponent=2.0,
          rms_min_waves=0.25, rms_max_waves=0.25, q=2, num_airy=4.0,
          dark_hole_iwa=2.0, dark_hole_owa=4.0,
          use_coronagraph=True, objective="log_contrast", diversity="defocus", defocus_rad=1.8138,
          include_command=True, flatten_obs=True, image_scale="log", log_floor=-10.0, log_ceil=-2.0,
          max_steps=4, action_scale=A)
env = CoronagraphEnv(**kw)
print("image_shape:", env.optics.image_shape, " dark-hole pixels:", int(env.optics.dark_hole_mask.sum()))
print("obs dim:", env.observation_space.shape, " action:", env.action_space.shape)
check_env(env, warn=True)
print("CORONA ENV CHECK PASSED")

env.optics.flatten_dm(); env.optics.clear_correction(); env.optics.clear_aberration()
print(f"unaberrated dark-hole floor: {env.optics.dark_hole_contrast():.3e}")

obs, info = env.reset(seed=4242)
print(f"obs image range [{obs[:2*16*16].min():.3f}, {obs[:2*16*16].max():.3f}]  start contrast {info['contrast']:.3e}")

# Expert: does flattening the 21-mode aberration dig a deep dark hole through the vortex?
expert = LeastSquaresExpert(env.optics, env.modal_basis)
c0s, cfs, reach = [], [], []
for e in range(10):
    _, info = env.reset(seed=9000 + e); c0s.append(info["contrast"])
    c = expert.correction(); reach.append(float(np.max(np.abs(c)) / A))
    env.optics.set_correction_modes(c); cfs.append(env.optics.dark_hole_contrast())
print(f"expert: start contrast median {np.median(c0s):.3e} -> corrected {np.median(cfs):.3e} "
      f"({np.median(c0s)/max(np.median(cfs),1e-300):.0f}x)")
print(f"max|expert coeff|/action_scale median {np.median(reach):.2f}, worst {np.max(reach):.2f}")
PY

# Short PPO smoke with the coronagraphic config (measures throughput + early movement).
python src/train_ppo.py \
    --total-timesteps 12000 \
    --num-envs 8 --n-steps 500 --batch-size 500 \
    --num-control-modes 21 --ideal-modal-correction --action-mode absolute \
    --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
    --objective log_contrast --dark-hole-iwa 2 --dark-hole-owa 4 \
    --diversity defocus --defocus-rad 1.8138 \
    --include-command --flatten-obs --image-scale log --log-floor -10 --log-ceil -2 \
    --q 2 --num-airy 4 \
    --pupil-pixels 256 --max-steps 4 --action-scale 3e-7 \
    --rms-min 0.25 --rms-max 0.25 \
    --learning-rate 1e-3 --lr-linear-decay --gamma 0 --ent-coef 0.0 \
    --target-kl 0 --log-std-init 0.0 \
    --log-dir logs/corona_sanity --tb-log-name corona_sanity \
    --eval-freq 4000

echo "CORONA SANITY DONE"
