# Project status — RL wavefront control (2026-06-21)

**Goal.** Train a PPO controller for coronagraphic wavefront control toward the 10⁻¹⁰
contrast regime. The operating point of interest is **space telescopes that start from
small residual aberrations** (post-coarse-correction, Strehl ≫ 0.9) — not large initial
wavefront errors.

This file summarizes what works, what doesn't, and why. Detailed per-experiment
write-ups live alongside it: [`RESULTS_paper_replication.md`](RESULTS_paper_replication.md),
[`RESULTS_coronagraph.md`](RESULTS_coronagraph.md), and the plan in
[`PLAN_replicate_gutierrez.md`](PLAN_replicate_gutierrez.md).

---

## ✅ Successes

### 1. Non-coronagraphic Strehl replication (Gutierrez et al. 2024)
Pure model-free PPO reaches **Strehl 0.988** (median 0.999, 90% > 0.99) on the
non-coronagraphic Zernike-correction task. The decisive settings: **γ = 0** (greedy
one-shot correction / contextual bandit), **16×16 images** (533-dim flattened input),
**4-step episodes with the previous command in the observation**, and **absolute
action**. Run `replicate_paper.slurm` → `logs/paper_replication/`.
Full detail in [`RESULTS_paper_replication.md`](RESULTS_paper_replication.md).

**Why it works:** the reward is **Strehl**, i.e. the PSF *peak* — a strong, always-
available signal that the network can sense at any wavefront quality.

### 2. Warm-start spine (coronagraphic)
LS-expert → behavior cloning → PPO fine-tune. BC reached **7.4×** contrast
deterministically; a 100k-step PPO fine-tune (`finetune.slurm`) pushed it to **12.9×**
(eval return +1.11), *surpassing* the 8.1× phase-conjugation expert. This
expert → BC → `--init-model` pattern is the spine to keep.

### 3. Floor diagnostics (`efc_floor.py`)
Pinned the hard geometric limit: with a 16×16 DM and a 12 λ/D OWA the controllable band
(N/2 = 8 λ/D) is narrower than the dark hole, capping any controller at ≈ −7.3 log₁₀.
Larger DMs reach −8.7 (24×24) / −9.8 (32×32). The **ideal Zernike corrector** makes the
LS expert exact — its flatten-the-wavefront floor for the 21-mode geometry is
**1.285e-10**, the reference target for the experiments below.

---

## ❌ Failures / negative results

### A. Coronagraphic model-free contrast plateau — a *sensing* limit
With the coronagraph on and sensing from the dark image, PPO learns ~20× (6.7e-3 →
3.1e-4) then freezes. The deterministic policy corrects in step 1 and does not refine on
steps 2–4: once the hole is at ~3e-4 the residual wavefront leaves no extractable signal
in the dark image. The 21 modes *can* reach 1.28e-10, so this is an information/sensing
limit, not control. Full write-up: [`RESULTS_coronagraph.md`](RESULTS_coronagraph.md).

### B. Contrast-as-reward is degenerate — it collapses to PSF wrecking *(this experiment, 2026-06-21)*

We tried rewarding the agent directly on **dark-hole contrast** instead of Strehl, in
two phases.

**B1 — Non-coronagraphic, three reward shapes.** Bare-PSF annulus contrast (2–4 λ/D),
rms 0.25 waves. Three shapes of the same objective:

| Run | Reward | Final contrast (median) | Final Strehl | vs. unaberrated floor (1.67e-3) |
|---|---|---|---|---|
| `logc` | `-log10(C)/10` | 7.19e-4 | **0.0038** | 0.4× |
| `negc` | `-C/0.01` | 5.94e-4 | **0.0033** | 0.4× |
| `diffc` | `log10(C_prev/C)` | 6.19e-4 | **0.0033** | 0.4× |
| *start* | — | 9.11e-3 | 0.344 | 5.4× |

All three drive the annulus *below* the diffraction floor by **smearing the Airy core
and scattering light out of the scoring region** — Strehl collapses from 0.34 to ~0.003.
The reward *shape* is irrelevant; the *objective quantity* (contrast mean) is the
problem. Eval: `eval_contrast_reward.slurm`; runs `logs/creward_{logc,negc,diffc}`.

**B2 — Coronagraphic, low-aberration sweep.** The natural fix was "turn the coronagraph
back on (no bright core to smear) and start from a near-perfect wavefront (the real
space-telescope regime)." It did **not** help. rms ∈ {0.02, 0.05, 0.10} waves, corona
on, log-scaled dark-image sensing, `-log10(C)/10` reward:

| Run | rms (waves) | Start contrast / Strehl | Final contrast / Strehl | vs. expert floor 1.285e-10 |
|---|---|---|---|---|
| `rms02` | 0.02 | 6.0e-5 / **0.984** | 2.38e-4 / **0.0018** | 1.9e6× |
| `rms05` | 0.05 | 3.7e-4 / 0.907 | 2.83e-4 / 0.0024 | 2.2e6× |
| `rms10` | 0.10 | 1.4e-3 / 0.718 | 2.42e-4 / 0.0023 | 1.9e6× |

The smoking gun: **rms02 starts at Strehl 0.984 / contrast 6.0e-5 — already excellent —
and the policy makes it *worse*** (Strehl → 0.002, contrast → 2.4e-4). All three runs
collapse to the **same** fixed point (~2.4e-4, Strehl ~0.002), and steps 1–4 are
**identical to 4 sig figs** → the policy ignores its observation and emits a **constant
command**. Runs `logs/corona_lowab_rms{02,05,10}` (cancelled at ~6.3M/10M steps; the eval
metric had already plateaued at 1.42 by 6M, and the behavior is a clean constant-output
collapse, so additional steps would not change the verdict). Train
`slurm/corona_lowab.slurm`; eval `slurm/eval_corona_lowab.slurm`.

#### Diagnosis — the two failures compound
The coronagraph does **not** prevent wrecking. It only removes the *on-axis* starlight;
a large fixed aberration still produces a ~2.4e-4 *mean* in the 2–4 λ/D annulus, and
that basin is **wide and robust to the input**. The correct "refine toward the floor"
solution requires *reading* a near-zero-signal dark image (failure A) — a narrow target
the optimizer never finds. So weak sensing → the policy can't learn the
input→correction map → it collapses to the easiest constant output, which for a
contrast-*mean* reward is a moderate scattered dark hole. **Strehl works precisely
because the PSF peak is a strong always-available signal; the contrast mean is not.**

---

## Implications & paths forward

Pure contrast-as-reward is **not learnable model-free** in this setup — abandoned.
Ranked next steps:

1. **Warm-start contrast fine-tune** from the BC/expert policy (the spine). Initializing
   inside the "refine" basin should keep the policy from collapsing to the wrecking
   attractor. Highest-confidence path.
2. **"Both" observation** — feed the agent the bright/unblocked PSF (strong sensing
   signal) *alongside* the coronagraphic dark image (nulling target), to break sensing
   limit A. (Separate `oth_*` job pipeline is exploring this.)
3. **Composite reward** that protects throughput: `-log10(C) - λ·(1-Strehl)` (or penalize
   core-intensity growth) to make wrecking impossible by construction. Useful as a
   confirming ablation even if (1)/(2) succeed first.
4. **Differential reward against a maintained good wavefront** rather than an absolute
   annulus mean.

---

## ⏳ In progress: PO4NCPA — model-based RL *(2026-06-21)*

Replicating Nousiainen et al., *A&A* 709, A267 (arXiv:2604.00993) — the model-based
successor to Gutierrez et al. that targets our exact sensing wall. Instead of
model-free PPO, it learns a **differentiable dynamics model** of the focal image
(ensemble of 5 small U-Nets) and trains the policy by **back-propagating the reward
through imagined rollouts** of that model, so the policy gets an analytic gradient of
predicted focal-plane light even where dark-image exploration gives PPO nothing.
Implementation: [`src/po4ncpa.py`](../src/po4ncpa.py); plan:
[`PLAN_po4ncpa.md`](PLAN_po4ncpa.md). Key ingredients: sequential phase diversity
(state = `(o_t, o_{t-1}, a_{t-1})`, the step-to-step command change is the implicit
probe — no defocus), cube-root residual preprocessing, reward `−‖o_{t+1}‖²`,
absolute action. New env hooks: `image_scale="raw"`, `diversity="none"`, `ideal_image()`.

**Milestone 1 — static NCPA, non-coronagraphic Strehl (correctness check): PASS —
reaches the paper's Strehl** once the setup was made paper-faithful.
`slurm/po4ncpa_strehl.slurm` → `logs/po4ncpa_strehl/`.

Paper-faithful config (`q=3`, num_airy 5.5 → 33×33; **55 Zernike modes from Noll 4,
i.e. piston+tip/tilt removed**; **Kolmogorov-like 1/f^(11/3) modal spectrum**; **0.026-wave
= 286 nm-equivalent RMS**; 4000-episode warm-up). Diagnostic (2500 ep) already reached
**Strehl 0.9935 and still rising** — at the paper's 0.994.

| | eval Strehl |
|---|---|
| no-op baseline (0.026 waves) | 0.974 |
| **PO4NCPA, converged 30k run (150 held-out ep)** | **0.9988** (best 0.9995, worst 0.9940; **100% > 0.99**) |
| ideal 55-mode correction floor | 1.000 |
| paper (static SI) | 0.994 |
| our PPO replication (Gutierrez) | 0.988 |

Beats the paper and our model-free PPO. Held-out eval (`slurm/eval_po4ncpa.slurm`) also
confirms **sequential phase diversity is active**: per-step Strehl climbs over the first
~4 steps (0.974 → 0.979 → 0.996 → 0.9985 → 0.999) then holds — not a one-shot
correction. The policy uses the temporal frame pair to converge, the paper's mechanism.

**The decisive fix was per-mode action scaling** — the paper's "actions scaled by max
expected NCPA per mode," which I had first implemented as a single scalar. The
sequence of failures and fixes is the real lesson here:

1. *Scalar action too large for the regime* → warmup explores mostly-wrecking commands,
   model is data-starved near the tiny optimum, policy collapses to PSF-wrecking.
   Partial fix: shrink the scalar (3e-7 → 1e-7); worked at 20 modes (plateau 0.975).
2. *55 modes with a scalar scale* → **catastrophic collapse** (Strehl → 0.011,
   monotonically worse over 30k episodes). High-order modes carry ~zero Kolmogorov
   power but get full authority to inject speckle, and are the hardest for the dynamics
   model to predict, so the policy exploits them.
3. **Fix:** per-mode action authority ≈ 5× each mode's expected NCPA RMS (Monte-Carlo
   calibrated via `optics.ideal_modal_correction_coeffs()`), so near-zero-power modes
   get near-zero stroke. → smooth monotonic climb to paper-level Strehl, no wrecking.

Other MBRL-specific safeguards in the implementation: random ensemble member per
imagined rollout step (curbs model-exploitation), cube-root residual reward, 4000-ep
dynamics-only warm-up. Full converged run (30k ep) in progress for the final number.

**Milestone 2 — static NCPA, coronagraph ON: PASS — breaks the ~3e-4 model-free PPO
plateau by ~21×, no wrecking.** (`slurm/po4ncpa_corona.slurm` → `logs/po4ncpa_corona/`,
same paper-faithful config + per-mode scaling, dark hole 1.5–4 λ/D in the 33×33 FoV.)
150 held-out episodes (`slurm/eval_po4ncpa.slurm po4ncpa_corona best 150`):

| | dark-hole contrast | Strehl |
|---|---|---|
| start (0.026 waves) | 1.78e-4 | 0.974 |
| old model-free PPO plateau | 3.1e-4 | (wrecks: 0.002 on contrast-reward) |
| **PO4NCPA, final (step 20)** | **1.49e-5** (best 2.9e-6, worst 4.7e-5) | **0.9976** (100% > 0.99) |
| PO4NCPA, best step (step 4) | 5.9e-6 | 0.9986 |
| ideal 55-mode correction floor | **1.18e-10** | 1.000 |

**~21× below the PPO plateau (best episode ~106×), while *holding* Strehl 0.998** — the
opposite of the contrast-reward collapse (failure B). The learned differentiable model
extracts far more dark-image signal than PPO, and the per-mode-scaled, gradient-trained
policy refines toward the null instead of collapsing to a constant wrecking command.

**The remaining gap is sensing-limited, not control-limited.** The ideal 55-mode
corrector reaches **1.18e-10** on the identical geometry/modes — so the control authority
to dig to 10⁻¹⁰ is present. PO4NCPA stops 5 orders of magnitude above it (ratio ~1.3e5×)
because at Strehl 0.998 the residual wavefront (σ ≈ 0.05 rad) scatters ~1.5e-5 speckle
and the dark image no longer carries enough signal to localize it. PO4NCPA pushes the
sensing wall ~21× deeper than PPO but does not eliminate it — consistent with the paper's
coronagraphic finding that focal-plane sensing, not control, is the binding constraint.

*Trajectory note:* per-step contrast deepens to 5.9e-6 at step 4 then relaxes to ~1.4e-5
and oscillates — the policy's best state is mid-episode and it drifts ~2.5× shallower by
step 20 (slight over-correction past its own optimum). Eval reports step-20, so the
headline understates achievable contrast; early-stopping / step-4 readout logs ~6e-6.
A small, fixable stability item, not a fundamental limit.

**Milestone 3 — dynamic NCPA (frozen-flow temporal disturbance): not started.**

---

## Repository layout (reorganized 2026-06-21)

```
RL/
├── src/        python modules: optics, environment, expert, pretrain_bc, train_ppo, efc_floor
├── slurm/      all .slurm batch scripts
├── notes/      markdown docs (this file, RESULTS_*, PLAN_*)
├── outputs/    slurm-<jobid>.out console logs
├── logs/       training runs (tensorboard + models/best_model.zip + checkpoints)
└── legacy/     archived earlier work
```

**Submitting jobs:** run `sbatch slurm/<script>.slurm [args]` **from the `RL/` root** so
that `outputs/slurm-%j.out` and the relative `logs/` paths resolve correctly. Each script
`cd`s to the repo root and exports `PYTHONPATH=…/RL/src` so both `python src/<module>.py`
and the inline `python - <<PY` eval heredocs import the package cleanly.
