#!/bin/bash
set -euo pipefail

# Submit the complete five-reward PO4NCPA ablation from the RL repository root.
# Usage: bash slurm/submit_reward_ablation.sh [RUN_TAG] [EPISODES]
# Example: bash slurm/submit_reward_ablation.sh retry_20260925 30000

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RL_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$RL_ROOT"

RUN_TAG=${1:-$(date +%Y%m%d_%H%M%S)}
EPISODES=${2:-30000}

case "$RUN_TAG" in
    *[!A-Za-z0-9._-]*) echo "RUN_TAG may contain only letters, numbers, dot, underscore, and dash" >&2; exit 2 ;;
esac
case "$EPISODES" in
    ''|*[!0-9]*) echo "EPISODES must be a positive integer" >&2; exit 2 ;;
    0) echo "EPISODES must be a positive integer" >&2; exit 2 ;;
esac

REWARDS=(
    log_energy
    sqrt_energy
    masked_log_energy
    masked_log_cubic
    masked_tail_log_cubic
)

echo "Submitting reward ablation: tag=$RUN_TAG episodes=$EPISODES"
for reward in "${REWARDS[@]}"; do
    job_id=$(sbatch --parsable --job-name="rw_${reward}" \
        slurm/po4ncpa_reward_screen.slurm "$reward" "$RUN_TAG" "$EPISODES")
    echo "  $reward: job $job_id -> logs/po4ncpa_reward_${reward}_${RUN_TAG}"
done

echo "Monitor with: squeue -u $USER"
