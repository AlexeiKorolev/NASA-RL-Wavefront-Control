#!/bin/bash
#SBATCH --job-name=rl_sanity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2
cd /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL
export PYTHONPATH=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/src:$PYTHONPATH

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== regression: optics floor (vortex_q=256 default) + env self-test ==="
python src/optics.py
python - <<'PY'
from stable_baselines3.common.env_checker import check_env
from environment import CoronagraphEnv
check_env(CoronagraphEnv(), warn=True)
print("ENV CHECK PASSED")
PY

echo "=== short real-config PPO run (8 envs) to confirm throughput ==="
python src/train_ppo.py --total-timesteps 4000 --num-envs 8 --eval-freq 100000 \
    --tb-log-name sanity --log-dir logs/_sanity
echo "SANITY OK"
