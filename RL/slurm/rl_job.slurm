#!/bin/bash
#SBATCH --job-name=rl_coro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # one core per parallel env (see --num-envs below)
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2
cd /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL
export PYTHONPATH=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/src:$PYTHONPATH

# Pin BLAS/FFT to one thread per process: each SubprocVecEnv worker is
# single-threaded, so throughput scales with --num-envs instead of fighting
# over cores (this is what turned ~1 step/s into ~100 steps/s).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python src/train_ppo.py \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --num-actuators-across 16 \
    --pupil-pixels 256 \
    --max-steps 20 \
    --action-scale 1e-8 \
    --rms-min 0.02 --rms-max 0.08 \
    --log-dir logs/ppo_1M_16x16 \
    --tb-log-name ppo_1M_16x16 \
    --eval-freq 20000
