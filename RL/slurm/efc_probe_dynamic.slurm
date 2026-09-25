#!/bin/bash
#SBATCH --job-name=efc_dyn_m4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# M4 baseline: classical pairwise-probe EFC vs DYNAMIC aberrations. The aberration takes
# one AR(1) drift step (constant RMS, Kolmogorov modal spectrum, correlation time tau in
# EFC iterations) before every sensing+correction iteration, so static EFC re-chases the
# disturbance each frame and its steady-state TRACKING contrast (mean of last 3 iters)
# rises as tau shrinks. tau=0 first as a continuity check (must reproduce 1.178e-10).
# This is the curve the learned probe policy (po4ncpa_probe_dynamic.slurm) must beat.
CKPT=${1:-logs/po4ncpa_corona/po4ncpa_best.pt}

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

for TAU in 0 100 30 10 3 1; do
    echo "=== EFC vs drift: tau=${TAU} ==="
    python src/efc_probe.py --checkpoint "${CKPT}" --n-draws 15 --efc-iters 20 \
        --tau ${TAU}
done

echo "=== EFC PROBE DYNAMIC M4 DONE ==="
