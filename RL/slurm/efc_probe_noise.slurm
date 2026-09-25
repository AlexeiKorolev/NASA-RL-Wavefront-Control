#!/bin/bash
#SBATCH --job-name=efc_noise_m3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# M3: classical pairwise-probe EFC vs photon budget. Sensing (the intensity-difference
# measurement) is shot-noise-limited at each flux; the probe calibration and the true-
# contrast metric stay noiseless. Shows how EFC's achievable floor RISES as photons run
# out -- the curve the learned noise-aware probe must beat. CPU, minutes.
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

# flux = photons at the unaberrated peak; a dark-hole pixel at contrast C gets C*flux
# photons. Sweep from near-noiseless (1e12) down to photon-starved (1e6), + noiseless ref.
python src/efc_probe.py --checkpoint "${CKPT}" --n-draws 15 --efc-iters 10 \
    --flux-list 0,1e12,1e11,1e10,1e9,1e8,1e7,1e6

echo "=== EFC PROBE NOISE M3 DONE ==="
