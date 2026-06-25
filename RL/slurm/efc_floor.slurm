#!/bin/bash
#SBATCH --job-name=efc_floor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2
cd /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL
export PYTHONPATH=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/src:$PYTHONPATH

# Let numpy/BLAS use a few threads here (single process, SVD + many propagations).
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4

# Deepest dark-hole contrast reachable under perfect-knowledge EFC, for 16/24/32
# DMs. Tells us whether -10 is physically reachable and what control authority
# (actuator count vs. the 12 lambda/D OWA) it needs.
python src/efc_floor.py \
    --num-actuators 16,24,32 \
    --pupil-pixels 256 \
    --dark-hole-iwa 3 --dark-hole-owa 12 \
    --n-aberrations 6 \
    --efc-iters 12 \
    --rms-min 0.02 --rms-max 0.08
