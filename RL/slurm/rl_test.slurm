#!/bin/bash
#SBATCH --job-name=rl_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # one core per parallel env (see --num-envs below)
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2
cd /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL
export PYTHONPATH=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/src:$PYTHONPATH

# Pin BLAS/FFT to one thread per process so throughput scales with --num-envs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Fast sanity: modal env builds and check_env passes before committing to training.
python - <<'PY'
from stable_baselines3.common.env_checker import check_env
from environment import CoronagraphEnv
env = CoronagraphEnv(num_control_modes=20)
print("modal action dim:", env.action_space.shape)
check_env(env, warn=True)
print("MODAL ENV CHECK PASSED")
PY

# Short real-config test: 20 Zernike modes instead of 256 raw actuators.
python src/train_ppo.py \
    --total-timesteps 50000 \
    --num-envs 8 \
    --num-actuators-across 16 \
    --num-control-modes 20 \
    --pupil-pixels 256 \
    --max-steps 20 \
    --action-scale 1e-8 \
    --rms-min 0.02 --rms-max 0.08 \
    --learning-rate 1e-4 \
    --log-std-init -1 \
    --log-dir logs/ppo_test_modal20_lr1e4 \
    --tb-log-name ppo_test_modal20_lr1e4 \
    --eval-freq 10000
