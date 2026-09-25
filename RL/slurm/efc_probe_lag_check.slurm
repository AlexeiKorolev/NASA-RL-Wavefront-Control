#!/bin/bash
#SBATCH --job-name=efc_lagcheck
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Fairness check for M4: does EFC's huge lead over the learned probe under drift (4.15e-8
# vs the RL policy's ~5e-5 plateau at tau=10) come from a real sensing/control advantage, or
# from a structural asymmetry -- EFC senses AFTER the drift and reacts within the SAME
# iteration (zero lag), while environment.py's step() evolves the aberration BEFORE applying
# an action computed from the PREVIOUS step's observation (always one full step stale)?
# --match-rl-lag reproduces the RL env's exact actuation ordering inside efc_loop. tau=0 is
# a control (lag matching should be inert with no drift); tau=10 is the real test.
module purge
module load anaconda3/2025.6
conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2
cd /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL
export PYTHONPATH=/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/src:$PYTHONPATH

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
set -eo pipefail

echo "=== tau=0, lag-matched (sanity: should reproduce 1.178e-10) ==="
python src/efc_probe.py --n-draws 15 --efc-iters 20 --tau 0 --match-rl-lag

echo "=== tau=10, PLAIN EFC (zero lag; reference: 4.15e-08) ==="
python src/efc_probe.py --n-draws 15 --efc-iters 20 --tau 10

echo "=== tau=10, LAG-MATCHED EFC (one-step stale, like the RL env) ==="
python src/efc_probe.py --n-draws 15 --efc-iters 20 --tau 10 --match-rl-lag

echo "=== EFC LAG-FAIRNESS CHECK DONE ==="
