#!/bin/bash
#SBATCH --job-name=paper_repl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # one core per parallel env (--num-envs 16)
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
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

# Faithful replication of Gutierrez et al. (2024) non-coronagraphic experiment:
#   16x16 phase-diversity images + prev command (533-dim) -> MLP (default tanh),
#   21 ideal Zernike modes, ABSOLUTE action, gamma=0 (greedy one-shot per step),
#   4-step episodes, 1/f^2 lambda/4 aberrations, reward -(1-SR)^(2/5),
#   lr 1e-3 linearly decayed, rollout 8000 (16 envs x 500), batch 500,
#   SB3 defaults otherwise (ent_coef 0, no target_kl). 10M steps. Target: Strehl > 0.99.
python src/train_ppo.py \
    --total-timesteps 10000000 \
    --num-envs 16 --n-steps 500 --batch-size 500 \
    --num-control-modes 21 --ideal-modal-correction --action-mode absolute \
    --num-aberration-modes 21 --aberration-spectrum power_law --psd-exponent 2.0 \
    --no-coronagraph --objective strehl --diversity defocus --defocus-rad 1.8138 \
    --include-command --flatten-obs --image-scale linear \
    --q 2 --num-airy 4 \
    --pupil-pixels 256 --max-steps 4 --action-scale 3e-7 \
    --rms-min 0.25 --rms-max 0.25 \
    --learning-rate 1e-3 --lr-linear-decay --gamma 0 --ent-coef 0.0 \
    --target-kl 0 --log-std-init 0.0 \
    --log-dir logs/paper_replication --tb-log-name paper_replication \
    --eval-freq 50000 --checkpoint-freq 1000000 --seed 0

echo "=== PAPER REPLICATION DONE ==="
