# RESULT: coronagraphic dark-hole control (sense from the dark image) — PLATEAU

**Date:** 2026-06-20. **Run:** `logs/corona_replication/` (job 3278153, cancelled at
8.9M steps — converged at 600k). **Model:** `logs/corona_replication/models/best_model.zip`.
**Script:** `replicate_corona.slurm`. **Eval:** `eval_corona.slurm`.

## Setup
The non-coronagraphic recipe (γ=0, 4-step, absolute action, 21 ideal Zernike modes,
1/f² λ/4 aberration, command-in-obs, MLP, lr 1e-3 decay, rollout 8000 / batch 500) with
the **coronagraph ON**. The agent **senses from the coronagraphic (dark) images**
(in-focus + defocus, log-scaled `[-10,-2]`), reward `= −log10(dark-hole contrast)/10`.
Dark hole IWA 2–OWA 4 λ/D (fits the 16×16 FoV); achievable floor (expert, flatten) **1.28e-10**.

## Result: learns ~20×, then plateaus 6 orders of magnitude short
Eval reward climbed 0.83 → **1.40 by ~600k steps, then flat at 1.40 for the remaining
8.3M steps**. Deterministic eval (150 held-out aberrations), median per-step contrast:

| step | median contrast |
|---|---|
| start | 6.7e-3 |
| after step 1 | **3.12e-4** |
| after step 2 | 3.12e-4 |
| after step 3 | 3.12e-4 |
| after step 4 | 3.12e-4 |

Policy final: median **3.1e-4** (best 2.4e-4, worst 3.7e-4). Expert floor 1.28e-10 →
**policy is 2.4×10⁶× above the achievable floor**; PPO captured ~1.3 of the ~7.7 reachable decades.

## Mechanism: a sensing limit, not a control limit
Steps 2–4 are **identical to 4 significant figures** — the policy applies its whole
correction in step 1 and then cannot refine. The 21 modes *can* reach 1.28e-10 (the
expert does), so control authority is not the issue. The problem is information: once the
dark hole is at ~3e-4, the *residual* wavefront produces no extractable signal in the dark
image, so each subsequent observation is uninformative and the policy repeats the same
command. (Froze at 600k of 9M steps — not a training-budget issue.)

This reproduces the paper's coronagraphic wall (Gutierrez et al. reached ~5×; we reached
~20×) and shows mechanistically *why*: focal-plane coronagraphic sensing is information-
starved at depth. Contrast with the non-coronagraphic run, where **bright-PSF** sensing
reveals the residual at every level and the agent refined to Strehl 0.99.

## Next step (untested)
Break the plateau by giving the agent the missing signal — **"both" observation**
(bright PSF + coronagraphic image): sense the wavefront from the bright channel (proven to
reach Strehl 0.99 ⇒ flatten ⇒ ~1.3e-10 here), null in the coronagraphic channel. Expected
to dig far below 3e-4. Alternative coronagraph-only route: pairwise-probe / EFC-style field
estimation.
