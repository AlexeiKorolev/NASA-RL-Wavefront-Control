#!/bin/bash
#SBATCH --job-name=po4dynscr2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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

nvidia-smi --query-gpu=name,memory.total --format=csv || true

# Continuation of po4ncpa_dynamic_scratch_t10 (job 3313473), which descended cleanly and
# monotonically for the full 40k episodes (2.0e-4 -> 4.68e-5, no plateau) -- unlike the
# warm-started attempt, which was flat the whole time. Same continuation pattern as the
# static logdeep2/logdeep3 chain: resume from this run's OWN checkpoint and its OWN saved
# replay buffer (auto-detected next to --resume; no domain shift here, so no forced-fresh
# buffer trick), picking up its decayed end-of-run LRs/exploration (dyn-lr 1e-3*0.3=3e-4,
# pol-lr 1e-4*0.3=3e-5, explore-std 0.02) and decaying gently further. Also switches to the
# lighter dyn-iters/pol-iters/batch settings the static campaign adopted from logdeep2 on,
# for cheaper updates over a long continued dig.
# Target: EFC's tau=10 tracking floor = 4.15e-8 (slurm/efc_probe_dynamic.slurm, job 3313341).
python src/po4ncpa_infogain.py \
    --run-name po4ncpa_dynamic_scratch2_t10 \
    --resume logs/po4ncpa_dynamic_scratch_t10/po4ncpa_infogain_best.pt \
    --aberration-tau 10 \
    --use-coronagraph true \
    --num-modes 55 --aberration-start-mode 4 --q 3 --num-airy 5.5 \
    --dark-hole-iwa 1.5 --dark-hole-owa 4.0 \
    --rms-min 0.026 --rms-max 0.026 \
    --aberration-spectrum power_law --psd-exponent 3.667 \
    --per-mode-scale true --action-headroom 5.0 --probe-frac 0.1 \
    --num-workers 14 --max-steps 8 --lambda-info 0.0 \
    --reward log --lr-final-frac 0.5 --explore-std 0.02 --explore-std-final 0.01 \
    --resume-refill 20000 --buffer-save-every 10000 \
    --total-episodes 40000 --warmup-episodes 0 \
    --eval-every 250 --eval-episodes 30 \
    --ensemble 5 --ch 64 --dyn-iters 2 --pol-iters 2 --batch 256 \
    --dyn-lr 3e-4 --pol-lr 3e-5 \
    --seed 12

echo "=== PO4NCPA DYNAMIC SCRATCH2 (continuation, tau=10) DONE ==="
