#!/bin/bash
#SBATCH --job-name=paper_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
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

# Deterministic end-of-episode Strehl of the replication policy over a held-out
# test set -- the number directly comparable to the paper's ">0.99".
python - <<'PY'
import numpy as np
from stable_baselines3 import PPO
from environment import CoronagraphEnv

kw = dict(num_control_modes=21, ideal_modal_correction=True, action_mode="absolute",
          num_aberration_modes=21, aberration_spectrum="power_law", psd_exponent=2.0,
          rms_min_waves=0.25, rms_max_waves=0.25, q=2, num_airy=4.0,
          use_coronagraph=False, objective="strehl", diversity="defocus", defocus_rad=1.8138,
          include_command=True, flatten_obs=True, image_scale="linear", max_steps=4, action_scale=3e-7)
env = CoronagraphEnv(**kw)
model = PPO.load("logs/paper_replication/models/best_model.zip", device="cpu")

per_step = [[], [], [], []]; starts = []
for e in range(200):
    obs, info = env.reset(seed=40000 + e); starts.append(info["strehl"])
    done = False; k = 0
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a); per_step[k].append(info["strehl"]); k += 1
        done = term or trunc
final = np.array(per_step[-1])
print(f"start Strehl       mean {np.mean(starts):.4f}")
for k in range(4):
    print(f"  after step {k+1}: mean Strehl {np.mean(per_step[k]):.4f}")
print(f"FINAL (step 4) Strehl: mean {final.mean():.4f}  median {np.median(final):.4f}  "
      f"min {final.min():.4f}")
print(f"fraction of episodes with final Strehl > 0.99: {np.mean(final > 0.99):.2f}")
print(f"fraction > 0.95: {np.mean(final > 0.95):.2f}")
PY
echo "PAPER EVAL DONE"
