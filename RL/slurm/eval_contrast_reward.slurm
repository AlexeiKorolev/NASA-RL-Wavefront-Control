#!/bin/bash
#SBATCH --job-name=creward_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Usage: sbatch eval_contrast_reward.slurm <run_name1> [run_name2 ...]
RUNS="${@:?at least one run name required}"

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

# Deterministic per-step bare-PSF dark-hole contrast AND Strehl over a held-out test
# set, for each contrast-reward run. Compares the contrast-trained policies against the
# Strehl-trained paper policy and the unaberrated floor.
RUNS="$RUNS" python - <<'PY'
import os, numpy as np
from stable_baselines3 import PPO
from environment import CoronagraphEnv

kw = dict(num_control_modes=21, ideal_modal_correction=True, action_mode="absolute",
          num_aberration_modes=21, aberration_spectrum="power_law", psd_exponent=2.0,
          rms_min_waves=0.25, rms_max_waves=0.25, q=2, num_airy=4.0,
          dark_hole_iwa=2.0, dark_hole_owa=4.0,
          use_coronagraph=False, objective="log_contrast", diversity="defocus", defocus_rad=1.8138,
          include_command=True, flatten_obs=True, image_scale="linear", max_steps=4, action_scale=3e-7)
env = CoronagraphEnv(**kw)

# Unaberrated bare-PSF annulus floor (same geometry).
env.optics.flatten_dm(); env.optics.clear_correction(); env.optics.clear_aberration()
floor = env.optics.dark_hole_contrast(
    intensity=env.optics.normalized_intensity(coronagraph=False, lyot=False))
print(f"unaberrated bare-PSF annulus floor: {floor:.3e}\n")

runs = os.environ["RUNS"].split()
for run in runs:
    path = f"logs/creward_{run}/models/best_model.zip"
    if not os.path.exists(path):
        path = f"logs/{run}/models/best_model.zip"   # allow full path-ish names
    model = PPO.load(path, device="cpu")
    cper = [[], [], [], []]; sper = [[], [], [], []]; cstart = []; sstart = []
    for e in range(200):
        obs, info = env.reset(seed=40000 + e); cstart.append(info["contrast"]); sstart.append(info["strehl"])
        done = False; k = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            cper[k].append(info["contrast"]); sper[k].append(info["strehl"]); k += 1
            done = term or trunc
    cfin = np.array(cper[-1]); sfin = np.array(sper[-1])
    print(f"### {run}  (model {path})")
    print(f"  start: contrast median {np.median(cstart):.3e}   Strehl mean {np.mean(sstart):.4f}")
    for k in range(4):
        print(f"  step {k+1}: contrast median {np.median(cper[k]):.3e}   Strehl mean {np.mean(sper[k]):.4f}")
    print(f"  FINAL contrast: median {np.median(cfin):.3e}  best {cfin.min():.3e}  worst {cfin.max():.3e}")
    print(f"  FINAL Strehl:   mean {sfin.mean():.4f}  median {np.median(sfin):.4f}  "
          f"frac>0.99 {np.mean(sfin>0.99):.2f}")
    print(f"  contrast/floor: {np.median(cfin)/floor:.1f}x\n")
PY
echo "CONTRAST REWARD EVAL DONE"
