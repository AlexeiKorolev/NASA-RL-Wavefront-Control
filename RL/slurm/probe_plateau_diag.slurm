#!/bin/bash
#SBATCH --job-name=plateau_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu

# Diagnose the learned-probe plateau: run the loaded policy far past its training horizon
# and track true contrast + correction dynamics per step (src/probe_plateau_diag.py).
CKPT=${1:-logs/po4ncpa_probe_r2/po4ncpa_probe_best.pt}
STEPS=${2:-40}

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

python src/probe_plateau_diag.py --checkpoint "${CKPT}" --steps "${STEPS}" --draws 20

echo "=== PROBE PLATEAU DIAG DONE ==="
