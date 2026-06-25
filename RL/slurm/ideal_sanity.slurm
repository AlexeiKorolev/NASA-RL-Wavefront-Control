#!/bin/bash
#SBATCH --job-name=ideal_sanity
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

# 1) Env builds + Gymnasium API check for the ideal-corrector action space.
python - <<'PY'
import numpy as np
from stable_baselines3.common.env_checker import check_env
from environment import CoronagraphEnv
from expert import LeastSquaresExpert

env = CoronagraphEnv(num_control_modes=20, ideal_modal_correction=True)
print("ideal-corrector action dim:", env.action_space.shape)
check_env(env, warn=True)
print("IDEAL ENV CHECK PASSED")

# 2) Correctness: the phase-conjugation expert applied one-shot through the ideal
#    corrector should drive contrast to ~the unaberrated floor (exact cancellation).
expert = LeastSquaresExpert(env.optics, env.modal_basis)
facs = []
for e in range(10):
    _, info = env.reset(seed=7000 + e)
    c0 = info["contrast"]
    env.optics.set_correction_modes(expert.correction())
    cf = env.optics.dark_hole_contrast()
    facs.append(c0 / max(cf, 1e-300))
    print(f"  ab {e}: {c0:.3e} -> {cf:.3e}  ({c0/max(cf,1e-300):.3e}x)")
print(f"IDEAL EXPERT median improvement {np.median(facs):.3e}x  "
      f"(floor ~{env.optics.dark_hole_contrast():.2e})")
PY

# 3) Tiny PPO smoke test on the ideal-corrector config (does the loop run + log?).
python src/train_ppo.py \
    --total-timesteps 4000 \
    --num-envs 2 \
    --num-control-modes 20 \
    --ideal-modal-correction \
    --pupil-pixels 256 \
    --max-steps 20 \
    --action-scale 1e-8 \
    --rms-min 0.02 --rms-max 0.08 \
    --learning-rate 3e-4 \
    --log-std-init -1 \
    --log-dir logs/ideal_sanity \
    --tb-log-name ideal_sanity \
    --eval-freq 2000

echo "IDEAL SANITY DONE"
