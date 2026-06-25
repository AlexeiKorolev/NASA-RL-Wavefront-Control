#!/bin/bash
#SBATCH --job-name=creward_sanity
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

# (1) Measure the bare-PSF dark-hole annulus contrast scale (no coronagraph): the
# start contrast (0.25-wave aberration) and the flat / unaberrated floor. These set
# a sensible linear_ceil for the neg_contrast reward and confirm the geometry fits.
python - <<'PY'
import numpy as np
from environment import CoronagraphEnv
kw = dict(num_control_modes=21, ideal_modal_correction=True, action_mode="absolute",
          num_aberration_modes=21, aberration_spectrum="power_law", psd_exponent=2.0,
          rms_min_waves=0.25, rms_max_waves=0.25, q=2, num_airy=4.0,
          dark_hole_iwa=2.0, dark_hole_owa=4.0,
          use_coronagraph=False, objective="log_contrast", diversity="defocus", defocus_rad=1.8138,
          include_command=True, flatten_obs=True, image_scale="linear", max_steps=4, action_scale=3e-7)
env = CoronagraphEnv(**kw)
print(f"dark-hole pixels: {int(env.optics.dark_hole_mask.sum())}")
starts, floors = [], []
for e in range(30):
    obs, info = env.reset(seed=40000 + e); starts.append(info["contrast"])
    env.optics.clear_correction(); env.optics.clear_aberration()
    floors.append(env.optics.dark_hole_contrast(
        intensity=env.optics.normalized_intensity(coronagraph=False, lyot=False)))
print(f"START bare-PSF annulus contrast: median {np.median(starts):.3e}  "
      f"min {np.min(starts):.3e}  max {np.max(starts):.3e}")
print(f"FLAT (unaberrated) annulus floor: median {np.median(floors):.3e}")
PY

# (2) Smoke each new reward objective for a few thousand steps to confirm no crash.
for OBJ in log_contrast neg_contrast contrast; do
  echo "=== SMOKE objective=$OBJ ==="
  python src/train_ppo.py \
      --total-timesteps 4000 \
      --num-envs 4 --n-steps 250 --batch-size 250 \
      --num-control-modes 21 --ideal-modal-correction --action-mode absolute \
      --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
      --no-coronagraph --objective $OBJ --dark-hole-iwa 2 --dark-hole-owa 4 \
      --diversity defocus --defocus-rad 1.8138 \
      --include-command --flatten-obs --image-scale linear --linear-ceil 0.01 \
      --q 2 --num-airy 4 \
      --pupil-pixels 256 --max-steps 4 --action-scale 3e-7 \
      --rms-min 0.25 --rms-max 0.25 \
      --learning-rate 1e-3 --lr-linear-decay --gamma 0 --ent-coef 0.0 \
      --target-kl 0 --log-std-init 0.0 \
      --log-dir logs/creward_sanity_$OBJ --tb-log-name smoke --eval-freq 2000 --seed 0
done
echo "=== CONTRAST REWARD SANITY DONE ==="
