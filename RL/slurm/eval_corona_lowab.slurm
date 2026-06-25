#!/bin/bash
#SBATCH --job-name=corona_lowab_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Usage: sbatch eval_corona_lowab.slurm <run1>:<rms1> [<run2>:<rms2> ...]
#   e.g. sbatch eval_corona_lowab.slurm rms02:0.02 rms05:0.05 rms10:0.10
PAIRS="${@:?at least one run:rms pair required}"

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

# Deterministic per-step dark-hole contrast AND Strehl of each low-aberration
# coronagraphic policy vs the expert floor, over a held-out test set. Strehl is
# reported to confirm the policy corrects the wavefront (no PSF wrecking) rather than
# merely nulling the scoring region.
PAIRS="$PAIRS" python - <<'PY'
import os, numpy as np
from stable_baselines3 import PPO
from environment import CoronagraphEnv
from expert import LeastSquaresExpert

def make_env(rms):
    kw = dict(num_control_modes=21, ideal_modal_correction=True, action_mode="absolute",
              num_aberration_modes=21, aberration_spectrum="power_law", psd_exponent=2.0,
              rms_min_waves=rms, rms_max_waves=rms, q=2, num_airy=4.0,
              dark_hole_iwa=2.0, dark_hole_owa=4.0,
              use_coronagraph=True, objective="log_contrast", diversity="defocus", defocus_rad=1.8138,
              include_command=True, flatten_obs=True, image_scale="log", log_floor=-10.0, log_ceil=-2.0,
              max_steps=4, action_scale=3e-7)
    return CoronagraphEnv(**kw)

for pair in os.environ["PAIRS"].split():
    run, rms = pair.split(":"); rms = float(rms)
    path = f"logs/corona_lowab_{run}/models/best_model.zip"
    env = make_env(rms)
    model = PPO.load(path, device="cpu")
    expert = LeastSquaresExpert(env.optics, env.modal_basis)
    cper = [[], [], [], []]; sper = [[], [], [], []]; cstart = []; sstart = []; experts = []
    for e in range(150):
        obs, info = env.reset(seed=40000 + e); cstart.append(info["contrast"]); sstart.append(info["strehl"])
        saved = env.optics.correction_coeffs.copy()
        env.optics.set_correction_modes(expert.correction()); experts.append(env.optics.dark_hole_contrast())
        env.optics.set_correction_modes(saved)
        done = False; k = 0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            cper[k].append(info["contrast"]); sper[k].append(info["strehl"]); k += 1
            done = term or trunc
    cfin = np.array(cper[-1]); sfin = np.array(sper[-1])
    print(f"### {run}  (rms={rms} waves, model {path})")
    print(f"  start: contrast median {np.median(cstart):.3e}   Strehl mean {np.mean(sstart):.4f}")
    for k in range(4):
        print(f"  step {k+1}: contrast median {np.median(cper[k]):.3e}   Strehl mean {np.mean(sper[k]):.4f}")
    print(f"  FINAL contrast: median {np.median(cfin):.3e}  best {cfin.min():.3e}  worst {cfin.max():.3e}")
    print(f"  FINAL Strehl:   mean {sfin.mean():.4f}  median {np.median(sfin):.4f}  frac>0.99 {np.mean(sfin>0.99):.2f}")
    print(f"  EXPERT floor (flatten): median {np.median(experts):.3e}")
    print(f"  policy/expert gap: {np.median(cfin)/max(np.median(experts),1e-300):.0f}x above floor\n")
PY
echo "CORONA LOW-AB EVAL DONE"
