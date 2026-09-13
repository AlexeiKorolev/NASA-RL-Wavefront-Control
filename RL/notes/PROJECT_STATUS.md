# Project status — RL wavefront control

---

## One-page summary (updated 2026-09-13)

**What the research has shown so far**, top to bottom. Detailed sections and the full
history follow below; the newest thread (learned active probing) is written up in
[`RESULTS_learned_probing.md`](RESULTS_learned_probing.md).

**1. Reproduced the prior state of the art.**
- Model-free PPO (Gutierrez et al. 2024): **Strehl 0.988** non-coronagraphically.
- Model-based RL — **PO4NCPA** (Nousiainen et al. 2026; learns a differentiable camera and
  back-props reward through imagined rollouts): reproduced and **matched/beat** the paper —
  Strehl **0.9988** non-coronagraphically; with the coronagraph on, digs to **~1.5e-5**
  contrast (~21× past plain-PPO) while *holding* Strehl.

**2. Diagnosed the real bottleneck — sensing, not control.** Passive coronagraphic control
plateaus at ~1.5e-5 while the same 55 modes can physically reach **1.18e-10**. The
sign-bearing signal in a deep null *shrinks as the null deepens*, so passive
sequential-diversity sensing runs out ~5 decades above the floor. (This is exactly the
"deeper dark hole" the paper named as future work.)

**3. Core contribution — learned active probing.** Two-headed policy `[probe, correction]`,
probe-difference image as observation, reward back-props `correction ← reconstruction ←
probe` end-to-end. Result: **1.5e-5 → 3.70e-8** (~**400×** past passive PO4NCPA), Strehl → ~1.0,
no wrecking, clean monotonic descent (still improving when stopped).

**4. Where a learned controller beats the classical incumbent** — the axes where a
static-perfect-model assumption breaks:
- **Model mismatch:** trained on a 20%-miscalibrated system it reaches **7.5e-8**
  (self-calibrates to the true system). A grey-box hybrid pinned to a nominal model did
  *worse* — lesson: let the policy learn the real system.
- **Dynamic drift:** the learned probe tracks a boiling disturbance; and the earlier
  "classical control wins by ~1100×" turned out to be a **baseline-fairness artifact** — a
  fair lag-matched comparison puts them in the same order of magnitude.

**5. Ruled out (negative results that matter).** Contrast-as-reward model-free → PSF-wrecking
(killed); observable-only info-gain auxiliary → strictly worse (dropped); photon noise → *not*
the RL-win regime (classical control is highly photon-robust).

**Honest framing.** Model-based RL + learned probing breaks the passive sensing plateau ~400×
and becomes competitive-or-better precisely when the model is wrong or the disturbance evolves.
In the clean, static, well-modeled regime the classical approach still wins, and RL does not
repeal the physics (broadband, picometer sensing, photon limits). Its case is being a better
*adaptive* controller — with sample efficiency / on-sky trainability as the open cost.

**Not yet closed (don't over-claim):** (a) the learned probe has **not reached the floor**
(3.70e-8, still descending); (b) the mismatch and dynamic wins are **single operating points on
still-training runs** — a gain-mismatch sweep and a τ-sweep are the natural next experiments.

---

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

**Next thesis — learned active probing (the highest-ceiling lever):** the passive dark
image is information-limited (5 decades above the floor). Plan +  progress in
[`PLAN_active_probing.md`](PLAN_active_probing.md).
- **M0 (done):** 8 modal probes reconstruct the complex dark-hole residual field to 0.68%
  at the plateau (4e-5) — the linear identity `I(+p)−I(−p)=4Re(E*E_p)` holds at corr 0.9999.
- **M1 (done) — pivotal:** classical pairwise-probe EFC, **intensity only**, reaches
  **1.178e-10** (the ideal floor) — **~127,000× below passive PO4NCPA's 1.5e-5**. *The
  sensing wall is not fundamental; passive PO4NCPA plateaued because it does not probe.*
  Consequently the *learned*-probe target (M2+) moves to the regimes where classical EFC
  degrades: no hand-built Jacobian, photon noise, dynamic NCPA, exposure efficiency.
- **M2 (done) — learned probe works, plateaus noiselessly:** the policy outputs
  `[probe, correction]`; a differentiable ensemble dynamics model predicts both the dark
  image and the probe-difference image, and the reward back-props into the probe
  (`src/po4ncpa_probe.py`). Descends **1.8e-4 → 1.39e-6** (best, ~36k episodes over two
  runs), Strehl 0.9995 — no PSF-wrecking, but **~4 orders above the EFC floor** and still
  decelerating. Mechanism validated; noiseless refinement is not the interesting regime
  (EFC already owns it).
- **Observable-only info-gain auxiliary (tried, negative — 2026-07-03):** to break the M2
  plateau *without* the illegal true-field label used by the recon variant
  (`po4ncpa_probe_recon.py` — unmeasurable in flight), added a reward that pays the probe
  for the dynamics model's *reliance on the difference channel* when predicting the next
  dark image — built entirely from observed intensity (`src/po4ncpa_infogain.py`). A λ
  sweep at identical config (`slurm/po4ncpa_infogain_sweep.slurm`, runs
  `logs/po4ncpa_infogain_l{0.0,0.1,0.3}`) is **decisive: λ=0 (pure control) descends
  cleanly to 1.44e-4 by ep 4k (S 0.98); λ>0 is strictly worse and unstable** (λ=0.1 wanders
  to 2.8e-4, λ=0.3 worse). The model barely uses input-`d` to predict `o'`, so the
  auxiliary injects a weak/noisy competing gradient. **Dropped.** Consolation prize: λ=0 is
  a reward-only probe running at **18× speed** (parallel collection + `max_steps=8`,
  benchmarked `slurm/bench_probe_speed.slurm`), now the clean fast baseline
  (`slurm/po4ncpa_probe_fast.slurm`).
- **M3 (done — photon noise does NOT favor RL, 2026-07-03):** env exposures are per-shot
  Poisson (`photon_flux`); `efc_probe.py --flux-list` (`slurm/efc_probe_noise.slurm`) runs
  probe-EFC with noisy sensing (noiseless calibration + true-contrast metric). Result:
  **EFC is highly photon-robust** — floor (1.178e-10) held to 1e10 ph/peak; 1.84e-10 @ 1e8;
  7.86e-10 @ 1e7; only **5.98e-9 @ 1e6** (51× degraded). Since the learned probe (M2) tops
  out at 1.39e-6, *maximally-degraded EFC is still ~230× better*; EFC only reaches 1.39e-6
  around an unphysical ~1e3 ph/peak. **Conclusion: photon noise is not the regime where the
  learned probe beats EFC** — the learned probe's own 4-order noiseless plateau is the
  binding constraint, not EFC's noise sensitivity. (EFC here is slightly optimistic: its
  regularization line-search uses oracle true-contrast; a noisy line-search would degrade it
  somewhat more, but not 4 orders.) The learned-probe-under-noise training was therefore not
  run — foregone. **Real RL-advantage regimes left: model error (miscalibrated/stale
  Jacobian) and dynamic NCPA (M4); and the prerequisite unlock is closing the learned
  probe's 4-order noiseless gap.**
- **Plateau attack — the 1.4e-6 wall was a REWARD artifact, now digging to 8.7e-8
  (2026-07-04):** a per-step diagnostic (`src/probe_plateau_diag.py`) showed the M2 policy
  converges in ~5 steps to a *fixed point* at 1.4e-6 (‖Δcorr‖→0), and 40 steps don't help —
  so it is not a step/iteration or geometry limit. Cause: the raw reward `−‖o′‖²` has
  vanishing gradient near the floor, so the policy stalls in its flat bottom. Fixes in
  `src/po4ncpa_infogain.py`: **log-contrast reward** (`--reward log`, ~const gradient per
  decade, like EFC's residual-proportional pull) + **LR decay** + **exploration decay** +
  a **resume refill-gate** (freeze loaded model/policy until the fresh buffer refills — kills
  the resume transient without persisting the ~3.6 GB buffer). Resuming the M2 best through
  two deep runs (`slurm/po4ncpa_probe_deep.slurm` raw→7.7e-7, then
  `slurm/po4ncpa_probe_logdeep.slurm` log→**8.7e-8**) gave a clean **monotonic** descent,
  Strehl *rising* to 0.99994 — **1.39e-6 → 8.7e-8, ~16× deeper, gap to EFC floor now ~740×
  (was ~11,800×).** Still descending at 40k episodes; `logdeep2` continues the dig. Also:
  the per-episode-cost win from `bench_probe_speed.slurm` (**18×**: 14-worker parallel
  collection + `max_steps=8`; the diagnostic justified 8 — policy converges by step 5) is
  what made this iteration affordable.
- **Campaign continued to 3.70e-8 (`logdeep3`, 2026-07-05):** the gentle continuation
  schedule (start at logdeep2's decayed LRs / low exploration, buffer save+reload) removed
  the resume sawtooth — clean immediate descent, Strehl 1.0000. Cumulative ~175k episodes;
  gap to the 1.18e-10 EFC floor now **314×**, still descending but decelerating (~halving
  per 40k eps). Runs: `logdeep`/`logdeep2`/`logdeep3`; figure `notebooks/figs/probe_campaign.png`.

### Why not EFC? — the model-mismatch experiment *(2026-07-06)*
EFC's power in our sim is entirely its **perfect analytic model**: we hand it the exact
control Jacobian + probe response, built from the same optics that generates the truth
(perfect-knowledge EFC 1.160e-10 ≈ intensity-only probe-EFC 1.178e-10 — its edge is the
model, not field access). On a real coronagraph that model is never exact. New knob
`optics.correction_gain` (per-mode corrector gain, default 1 → every prior run unchanged;
`set_correction_gain`) makes the truth realize `gain·coeff` while a nominal-gain controller
is miscalibrated — the canonical actuator-gain uncertainty.

**Phase 1 — EFC degradation sweep (`src/efc_mismatch.py`, `slurm/efc_mismatch.slurm`, job
3296740, DONE).** Builds EFC's model at gain=1, runs the loop on a gain-mismatched truth,
sweeps σ. σ=0 reproduces **1.178e-10 exactly** (mismatch plumbing provably inert at unit
gain). Then it craters:

| gain mismatch σ | perfect-sensing EFC | probe-EFC | vs ideal floor |
|---|---|---|---|
| 0% | 1.160e-10 | 1.178e-10 | 1.0× |
| 2% | 1.19e-9 | 1.21e-9 | 10× |
| 5% | 3.34e-8 | 3.02e-8 | 257× |
| 10% | 3.16e-8 | 3.19e-8 | 270× |
| 20% | 1.58e-7 | 2.08e-7 | 1765× |
| 30% | 4.34e-7 | 4.49e-7 | 3807× |

Figure `notebooks/figs/efc_mismatch.png`. **Honest twist (falsified my hypothesis):** I
predicted the damage would come through *sensing* (biased probe reconstruction) and that
perfect-field EFC would stay near the floor. It does **not** — both curves degrade nearly
identically, so the damage is dominated by **control**: a per-mode gain rotates the
commanded correction in modal space (truth applies `gain·dc`, not `dc`), which a scalar
step / rcond line-search cannot undo, so even perfect field knowledge converges to a
residual floor. The point is stronger for it — EFC's *entire* operation rests on a static
model, and when that model is wrong neither iteration nor perfect field access rescues it.
The learned probe (3.70e-8 on the true system) crosses over and **beats realistic EFC past
~10% mismatch** — the realistic pre-flight / post-drift band. Below ~5% a well-calibrated
EFC still wins: RL is a scalpel, not a silver bullet.

**Phase 2 — RL under mismatch + grey-box hybrid: DONE (2026-07-07), pure RL wins.**
`ProbeConfig` gains `correction_gain_sigma/seed` (RL trains+evals on the same gained truth
→ learns the true system) and `efc_warmstart`/`efc_warmstart_iters` (new `EFCWarmStartEnv`:
each reset drives the corrector to EFC's converged floor, then the RL policy learns the
**residual** EFC's miscalibrated model can't reach). Both warm-start from `logdeep3` with a
**fresh** buffer. Scripts: `slurm/po4ncpa_probe_mismatch.slurm [σ]`,
`slurm/po4ncpa_probe_hybrid.slurm [σ]`, smoke `slurm/po4ncpa_mismatch_smoke.slurm`.

Results at σ=0.20 (probe-EFC there: 2.08e-7):

| controller @ σ=0.20 | eval contrast | Strehl |
|---|---|---|
| probe-EFC (miscalibrated model) | 2.08e-7 | — |
| **pure RL** (`logs/po4ncpa_mismatch_s0.20`, job 3297079, 40k ep) | **7.5e-8** | 1.0000 |
| EFC-warm-start hybrid (`logs/po4ncpa_hybrid_s0.20`, job 3297104, 25k ep) | 2.6e-7 | 0.9966 |

**Pure RL beats mismatched EFC ~2.8× and — surprise — beats the hybrid too:** the hybrid
plateaus *at* EFC's miscalibrated floor. Diagnosis: the EFC warm-start pins every episode
to the biased base command, so the residual policy inherits EFC's model error instead of
escaping it; the pure policy, free to learn the true gained system end-to-end, digs past
it. The grey-box "predict the residual" idea is dead on this axis; pure RL on the true
system is the winner under model mismatch.

### M4 — dynamic aberrations (in progress, 2026-07-19)
The last RL-advantage axis from `PLAN_active_probing.md`: a temporally evolving disturbance
that static EFC must re-chase each frame while the learned probe policy can predict ahead.
New machinery (defaults off, zero risk to prior runs):
- `optics.set_random_aberration` now stores its modal coeffs; new
  `optics.evolve_aberration(rho, rng)` takes one AR(1) step `c ← ρc + √(1−ρ²)·ξ` with the
  same Kolmogorov modal spectrum, renormalized to constant RMS ("boiling" at fixed WFE).
- `CoronagraphEnv(aberration_tau=…)`: correlation time in control steps (ρ = exp(−1/τ));
  the drift step fires at the top of `step()`, so each action acts on a field one frame
  staler than its observation. `ProbeConfig.aberration_tau` plumbs it through the trainer.
- `efc_probe.py --tau`: drifts the truth between EFC iterations; reports steady-state
  **tracking** contrast (mean of last 3 iters), same drift realization across sensing modes.
Scripts: `slurm/efc_probe_dynamic.slurm` (τ ∈ {0,100,30,10,3,1} baseline sweep, job
3313341), `slurm/po4ncpa_dynamic_smoke.slurm` (job 3313340, smoke-tested clean).

**EFC drift baseline: DONE (2026-07-19).** τ=0 reproduces the static floor exactly
(1.178e-10 vs the M1 reference 1.178e-10) — continuity check passes. Tracking contrast
(mean of the last 3 of 20 iterations) degrades smoothly as τ shortens, roughly τ⁻²
(one-frame lag error ∝ drift step, contrast ∝ error²):

| τ (control frames) | EFC tracking contrast |
|---|---|
| 0 (static) | 1.178e-10 |
| 100 | 6.59e-10 |
| 30 | 4.66e-9 |
| 10 | **4.15e-8** |
| 3 | 4.71e-7 |
| 1 | 4.48e-6 |

**Learned-probe attempt #1 — warm-started, STALLED (killed).** `slurm/po4ncpa_probe_dynamic.slurm`
resumed from the static logdeep3 checkpoint with a fresh buffer (job 3313383, `logs/po4ncpa_dynamic_t10`).
Eval contrast sat flat at 3.5–3.9e-5 for 17k+ episodes with zero downward trend — indistinguishable
from the smoke test's *un-retrained* static policy under drift (~4.2e-5). Diagnosis: the dynamics
net's forward signature is `(obs, prev_action, action) -> next_image`, a deterministic mapping fit
entirely on zero-drift data; the warm start anchored both nets near that static-world fit instead of
letting them learn the drift-injected residual energy. Killed at ep ~17.4k to free the GPU.

**Learned-probe attempt #2 — FROM SCRATCH: clean monotonic descent, still going.**
`slurm/po4ncpa_probe_dynamic_scratch.slurm` (job 3313473, `logs/po4ncpa_dynamic_scratch_t10`,
40k ep, no `--resume`, `reward=log` from the start, proper 4000-ep warmup restored). Unlike
attempt #1, this **never stalled**:

| episode | eval contrast |
|---|---|
| 4032 (warmup ends) | 2.01e-4 |
| 12096 | 1.48e-4 |
| 20160 | 1.04e-4 |
| 26208 | 6.89e-5 |
| 32256 | 5.54e-5 |
| 38304 | 4.86e-5 |
| **40000 (final)** | **4.68e-5** |

Monotonic the whole run, no plateau — same shape as the static M2 campaign's first 40k-episode
run (landed at 1.39e-6, then the logdeep/logdeep2/logdeep3 continuation chain dug another 4
orders of magnitude to 3.70e-8). Still **~1100× above the EFC τ=10 tracking floor (4.15e-8)**.
Continuation launched (`slurm/po4ncpa_probe_dynamic_scratch2.slurm`, job 3313539,
`logs/po4ncpa_dynamic_scratch2_t10`): resumes from this run's own checkpoint + own saved
replay buffer (no domain shift this time, unlike the mismatch runs' forced-fresh-buffer trick),
picks up the decayed end-of-run LRs (dyn 3e-4, pol 3e-5, explore 0.02) and the lighter
dyn-iters/pol-iters/batch settings the static campaign used from logdeep2 onward, decaying
further. Target: descend toward or past 4.15e-8. Result pending.

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
