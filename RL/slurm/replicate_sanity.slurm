#!/bin/bash
#SBATCH --job-name=replicate_sanity
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

# Phase 0 sanity for the Gutierrez non-coronagraphic replication (see PLAN_replicate_gutierrez.md):
#   no coronagraph, ideal 21-mode Zernike corrector, 1/f^2 aberration at lambda/4 RMS,
#   in-focus + defocused PSF observation + previous command, Strehl objective.
python - <<'PY'
import numpy as np
from stable_baselines3.common.env_checker import check_env
from environment import CoronagraphEnv
from expert import LeastSquaresExpert

kw = dict(num_control_modes=21, ideal_modal_correction=True,
          num_aberration_modes=21, aberration_spectrum="power_law", psd_exponent=2.0,
          rms_min_waves=0.25, rms_max_waves=0.25,
          use_coronagraph=False, objective="strehl", diversity="defocus", defocus_rad=1.0,
          include_command=True, max_steps=8, action_scale=5e-8)

env = CoronagraphEnv(**kw)
print("obs space:", env.observation_space)
print("action dim:", env.action_space.shape)
check_env(env, warn=True)
print("REPLICATE ENV CHECK PASSED")

# Observation ranges (image channels + command vector) on a fresh aberration.
obs, info = env.reset(seed=4242)
img, cmd = obs["image"], obs["command"]
print(f"image range [{img.min():.3f}, {img.max():.3f}]  shape {img.shape}")
print(f"command range [{cmd.min():.3f}, {cmd.max():.3f}]  shape {cmd.shape}")
print(f"start Strehl {info['strehl']:.4f}")

# Correctness: the expert one-shot through the ideal corrector should reach Strehl ~1.0.
expert = LeastSquaresExpert(env.optics, env.modal_basis)
ss = []
for e in range(10):
    _, info = env.reset(seed=8000 + e)
    s0 = info["strehl"]
    env.optics.set_correction_modes(expert.correction())
    sf = float(np.max(env.optics.normalized_intensity(coronagraph=False, lyot=False)))
    ss.append(sf)
    print(f"  ab {e}: Strehl {s0:.4f} -> {sf:.4f}")
print(f"EXPERT median corrected Strehl {np.median(ss):.4f}  (target > 0.99)")
PY

# Short PPO smoke test on the full replication config (MultiInputPolicy via --include-command).
python src/train_ppo.py \
    --total-timesteps 12000 \
    --num-envs 2 \
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
    --log-dir logs/replicate_sanity \
    --tb-log-name replicate_sanity \
    --eval-freq 4000

echo "REPLICATE SANITY DONE"
