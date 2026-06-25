# RESULT: Gutierrez et al. (2024) non-coronagraphic replication — SUCCESS

**Date:** 2026-06-19. **Run:** `logs/paper_replication/` (job 3277618, 10M steps).
**Model:** `logs/paper_replication/models/best_model.zip`.
**Script:** `replicate_paper.slurm` (exact command preserved there).

## Result
Pure PPO (no behavior cloning) learned to correct λ/4 aberrations to near-perfect
Strehl, reproducing [Gutierrez et al. 2024](https://arxiv.org/abs/2407.18733).

Deterministic eval, 200 held-out aberrations, per-step Strehl:

| step | mean Strehl |
|---|---|
| start | 0.344 |
| 1 | 0.602 |
| 2 | 0.864 |
| 3 | 0.963 |
| **4 (final)** | **0.988** |

Final Strehl: **mean 0.988, median 0.999, 90% of episodes > 0.99, 95% > 0.95** (min 0.47
on the hardest draws). Training eval reward climbed monotonically −3.62 → −1.17 over 10M
steps; this was the first run that learned at all (all prior runs flatlined or diverged).

## The exact working config (what finally matched the paper)
The breakthrough was matching the paper precisely. The decisive differences from our
earlier failed attempts:

| Parameter | Value | Why it mattered |
|---|---|---|
| **Discount γ** | **0** | each step is a greedy one-shot correction ("single-step" behavior) |
| **Image size** | **16×16** (`q=2, num_airy=4`) | 533-dim input; our 56×56 (6272-dim) was too large for the MLP |
| **Episode** | **4 steps + previous command in obs** | lets the greedy policy iteratively converge |
| **Action** | **absolute** (`action_mode=absolute`) = the DM command, not an increment | matches paper's `action = φ_DM` |
| Corrector | ideal 21-mode Zernike (`ideal_modal_correction`) | `N_act = N_modes`, exact |
| Aberration | 21 Zernike modes, 1/f² PSD, λ/4 (0.25 wave) RMS | `aberration_spectrum=power_law` |
| Observation | in-focus + defocus PSF (λ/(2√3)≈1.81 rad) + command, flattened, linear-scaled | `flatten_obs, include_command, image_scale=linear, diversity=defocus` |
| Network | default SB3 tanh MLP | "defaults kept" |
| Reward | −(1−SR)^(2/5) | `objective=strehl` |
| LR | 1e-3, linear decay | `--lr-linear-decay` |
| Rollout / batch | 8000 (16 envs × 500) / 500 | paper values |
| ent_coef / target_kl | 0 / disabled | SB3 defaults |
| Steps | 10,000,000 | paper budget |

## Caveats / not-yet-matched
- **Detector noise OFF** for this run (paper used SNR≈100). Adding `photon_flux` to confirm
  Strehl holds is the remaining fidelity step.
- Paper did not specify the **action scaling** (chose `action_scale=3e-7`; sanity confirmed
  the expert correction is reachable at |a|≤0.58) or exact **network width** (used SB3 default).

## Dead ends (do not repeat)
- Coronagraphic model-free PPO from scratch → stuck near 0.
- Single-step (`max_steps=1`) pure PPO with large action_scale → diverged to Strehl ~0.
- 56×56 CNN with γ=0.99 → dead critic (explained_variance ≈ 0), no learning.
- BC single-step alone reached Strehl 0.79 (a useful fallback, but the paper config beats it).
