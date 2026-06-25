#!/bin/bash
#SBATCH --job-name=corona_eval
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

# Deterministic per-step dark-hole contrast of the coronagraphic policy vs the
# expert floor (~1.3e-10), over a held-out test set.
python - <<'PY'
import numpy as np
from stable_baselines3 import PPO
from environment import CoronagraphEnv
from expert import LeastSquaresExpert

kw = dict(num_control_modes=21, ideal_modal_correction=True, action_mode="absolute",
          num_aberration_modes=21, aberration_spectrum="power_law", psd_exponent=2.0,
          rms_min_waves=0.25, rms_max_waves=0.25, q=2, num_airy=4.0,
          dark_hole_iwa=2.0, dark_hole_owa=4.0,
          use_coronagraph=True, objective="log_contrast", diversity="defocus", defocus_rad=1.8138,
          include_command=True, flatten_obs=True, image_scale="log", log_floor=-10.0, log_ceil=-2.0,
          max_steps=4, action_scale=3e-7)
env = CoronagraphEnv(**kw)
model = PPO.load("logs/corona_replication/models/best_model.zip", device="cpu")
expert = LeastSquaresExpert(env.optics, env.modal_basis)

per_step = [[], [], [], []]; starts = []; experts = []
for e in range(150):
    obs, info = env.reset(seed=40000 + e); starts.append(info["contrast"])
    # expert floor for this aberration
    saved = env.optics.correction_coeffs.copy()
    env.optics.set_correction_modes(expert.correction()); experts.append(env.optics.dark_hole_contrast())
    env.optics.set_correction_modes(saved)
    done = False; k = 0
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a); per_step[k].append(info["contrast"]); k += 1
        done = term or trunc
import numpy as np
print(f"start contrast       median {np.median(starts):.3e}")
for k in range(4):
    print(f"  after step {k+1}: median contrast {np.median(per_step[k]):.3e}")
fin = np.array(per_step[-1])
print(f"POLICY final contrast: median {np.median(fin):.3e}  best {fin.min():.3e}  worst {fin.max():.3e}")
print(f"EXPERT floor (flatten): median {np.median(experts):.3e}")
print(f"policy/expert gap: {np.median(fin)/max(np.median(experts),1e-300):.0f}x above the achievable floor")
PY
echo "CORONA EVAL DONE"
