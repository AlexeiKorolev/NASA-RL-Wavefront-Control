# PLAN — learned active probing (model-based) to break the sensing wall

**Status:** proposed (2026-06-26). Prereq: PO4NCPA static milestones done
([`PROJECT_STATUS.md`](PROJECT_STATUS.md)). This is the "highest-ceiling lever."

## The problem, restated

The coronagraph camera measures **intensity only**, `I = |E|²`. At Strehl 0.998 the
residual dark-hole field `E` is tiny, so `I` is *quadratically* small and **sign-blind** —
`+δ` and `−δ` of aberration make the same speckle. PO4NCPA's passive policy therefore
plateaus at ~1.5e-5 (contrast), 5 decades above the 1.18e-10 floor it *could* reach with
the same 55 modes. The gap is **information**, not control. Measured, not hypothesized —
see `notebooks/figs/view2_trajectory.png`.

## The idea

Inject a *known* probe command `p` and read the **difference** of two exposures:

```
I(+p) − I(−p) = 4·Re(E* · E_p)          # LINEAR in the unknown field E, carries its phase
```

Two independent probes → real+imag of `E` → the exact nulling command. This is classical
pairwise-probing + electric-field conjugation (EFC), how real testbeds (HiCAT, Roman CGI)
reach 1e-8–1e-10. **Our novelty:** the probe sequence and the reconstruction are *learned
end-to-end* by the model-based agent, not hand-derived — and adapt as the field evolves.

## The key enabling observation

PO4NCPA's `DynamicsNet` already predicts the dark image produced by an applied command
delta from the current state. **That is exactly the field-response operator a probe needs.**
So the same trained model, queried at `+p` and `−p`, yields a *predicted* difference image;
backprop of the downstream contrast reward flows correction ← reconstruction ← probe, so
**the probe is automatically shaped to be maximally informative for nulling.** Classical
EFC has no such end-to-end signal. Minimal new architecture; mostly new plumbing + heads.

## Control loop (per step `t`)

1. State: corrector coeffs `c_t`; unknown residual field `E_t` in the dark hole.
2. Policy proposes a **probe** `p_t` (learned).
3. Env takes two extra exposures: `I⁺ = img(c_t + p_t)`, `I⁻ = img(c_t − p_t)`;
   difference `d_t = I⁺ − I⁻`. (Plus the un-probed residual image `I⁰` as today.)
4. Observation → policy: `{cbrt(I⁰ − I_ideal), d_t, c_t}`.
5. Policy proposes **correction** `Δc_t` (learned); apply `c_{t+1} = c_t + Δc_t`.
6. Reward `r = −‖residual(c_{t+1})‖²` − `λ_probe·(probe cost)` (see M3).

Probe `p` and correction `Δc` both live in the existing ideal-corrector modal space, so a
probe phase is `extra_phase = _correction_modes.T @ p` — already supported by
`normalized_intensity(extra_phase=...)`. No DM/optics rewrite.

## Code touch-points

**`src/optics.py`**
- `focal_field(coronagraph, lyot, extra_phase=None) -> complex ndarray` — return the complex
  focal field (not just `.intensity`) for ground-truth reconstruction error + the aux loss.
  One-liner off the existing `_focal_intensity` path (`prop.forward(wf)` before `.intensity`).
- Probe exposures already work via `normalized_intensity(extra_phase=…, flux=…)`.

**`src/environment.py`** (new flags, default off → zero risk to current runs)
- `probe_mode: bool=False`. When on:
  - action = `concat(probe_coeffs, correction_coeffs)` (2×`num_modes`), both in [−1,1],
    probe scaled by its own per-mode `probe_scale`.
  - `step()` takes the ±probe exposures, builds `d_t`, and returns obs
    `{"image": stack([cbrt_resid, d_t]), "command": c_t}` (difference channel added).
  - expose `true_field()` (calls `optics.focal_field`) in `info` for eval/aux only.
- `probe_scale` calibrated like `calibrate_per_mode_scale` but sized to the *linear-regime*
  probe amplitude (a few × the expected residual, small enough to stay linear).

**`src/po4ncpa.py`**
- `PolicyNet`: second head → `(probe, correction)` (tanh each). Input gains the `d_t` channel.
- `DynamicsNet`: unchanged in form; it already maps (state, command) → image. In the
  imagined rollout, query it at `±p` to synthesize `d̂_t`, feed `d̂_t` to the policy to get
  `Δc`, query at `Δc` for the next residual → reward. Probe gets gradient through this chain.
- Optional `FieldHead` (small MLP/conv): `d_t → Ê_t` (complex dark-hole field), supervised
  by `optics.focal_field()` (Phase-M0 diagnostic; keep or drop after M2).
- Reward adds `−λ_probe·‖p_t‖²` (or a per-step photon penalty) once noise is on.

## Milestone ladder

**M0 — plumbing + linearization check (noiseless): PASS (2026-06-26).** Added
`optics.focal_field()` + `src/probe_check.py` (`slurm/probe_check.slurm`). In the plateau
regime (median contrast 4.0e-5), 8 modal probes reconstruct the complex dark-hole residual
field to **0.68%** per pixel, with the linear identity `I(+p)−I(−p)=4·Re(E*·E_p)` correlating
at **0.9999**. Reconstruction error rises with probe amplitude (0.68%→8.1% over 0.05→0.4×
per-mode scale), so the linear regime is **≈0.05–0.1× scale** — the probe amplitude to use
in M1/M2. Verdict: the signal the passive policy can't see is fully recoverable by probing —
the sensing wall is breakable. (Job 3293277.)

**M1 — classical pairwise-probe EFC baseline: DONE (2026-06-26) — reaches the floor.**
`src/efc_probe.py` (`slurm/efc_probe.slurm`). Pairwise-probe EFC in the 55-mode corrector
space, **intensity measurements only**, 15 held-out draws: median final contrast **1.178e-10**
— identical to perfect-knowledge EFC (1.160e-10) and the ideal floor (1.18e-10), i.e.
**~127,000× below passive PO4NCPA's 1.5e-5**, converging in ~4–5 iterations. (Job 3293278.)

**This is the pivotal finding: the sensing wall is not fundamental — passive PO4NCPA
plateaued because it does not probe.** With probing, intensity-only control reaches the
physical floor. One draw stalled at 4.8e-8 for *both* perfect and probe EFC → a
control/regularization conditioning issue on that aberration, not sensing (revisit).

**Consequence — M2's target moves.** In the idealized noiseless/static/perfect-Jacobian
case, classical EFC is already optimal; a learned probe cannot beat the floor here. So M2/M3
must justify learning on the axes where EFC degrades (see revised milestones): photon noise,
dynamic NCPA, model mismatch, and exposure efficiency (floor in fewer probes/photons).

**M2 — learned probe + learned reconstruction (the thesis, ~1 GPU run).** Two-headed policy,
difference-image observation, model-based training as above. Ablate: passive PO4NCPA (today)
vs fixed-random probe vs **learned probe**. Given M1 (classical EFC already reaches the floor
noiselessly), the M2 bar is *not* the noiseless floor — it is: **reach the floor without a
hand-built Jacobian** (the model-based agent learns the field-response operator itself), and
do it in **fewer probes/exposures** than fixed EFC (learned probes chosen for information gain
should beat fixed low-order probes). Metric: held-out contrast vs floor + probes-to-floor +
reconstruction error. This is the setup piece; the *payoff* is M3/M4.

*Implementation (2026-06-26):* `src/po4ncpa_probe.py` + env `probe_mode` (action=[probe(M),
correction(M)], obs=[dark image, probe-difference image], 2-channel dynamics predicts both,
reward backprops into the probe via the multi-step imagined rollout). Sanity smoke test
(job 3293280) ran clean end-to-end on GPU (shapes, dynamics train, policy gradient flows).
Full corona-aligned run `logs/po4ncpa_probe/` (job 3293541, 20k episodes, `slurm/po4ncpa_probe.slurm`).
Per-step Strehl propagation skipped except on the last step for speed.

**M3 — photon noise, where probing has to be smart (~1 GPU run).** Turn on `flux`
(SNR≈100). Now probe amplitude/count trades information against photons/time; add the probe
cost to the reward. This is the regime where *learned* probing should beat fixed EFC —
adapting probe strength to the local SNR. Report contrast(SNR) and probes-per-decade.

**M4 — dynamic NCPA (frozen-flow).** Probe under a temporally evolving disturbance; the
learned probe policy should exploit temporal correlation (predict-ahead) that static EFC
can't. Ties into the separately-planned dynamic-NCPA milestone.

## Risks / open questions

- **Probe nonlinearity:** amplitude too large breaks the linear difference identity; too
  small drowns in noise. Handled by learning per-mode `probe_scale`, but M0 must bound the
  linear regime.
- **Photon budget accounting:** each probe pair is 2 exposures. The honest figure of merit
  is contrast *per unit exposure/photons*, not per control step — bake into M3's reward.
- **Explicit vs implicit reconstruction:** the `FieldHead` (explicit `Ê`) aids interpretability
  and gives a supervised anchor, but the policy may do better mapping `d_t → Δc` implicitly.
  Keep the head through M2 as a diagnostic; decide by ablation whether to keep it.
- **Credit assignment to the probe:** relies on backprop through the dynamics model being
  faithful at probe scale; if the model is inaccurate for small `p`, add a supervised
  difference-image loss (`d̂` vs real `d`) — cheap, we already collect real probe exposures.

## Definition of done

M1 already proved the wall is breakable: intensity-only pairwise-probe EFC reaches the
1.18e-10 floor (~127,000× below passive PO4NCPA). So "done" is no longer about the noiseless
static floor — it is demonstrating that **learned** active probing wins where classical EFC
can't:

1. reaches the floor **without a hand-built Jacobian** (model-based agent learns the
   field-response operator), in **fewer exposures** than fixed-probe EFC (M2);
2. under **photon noise**, holds deeper contrast per photon than fixed EFC by adapting probe
   amplitude/count to local SNR (M3);
3. under **dynamic NCPA**, exploits temporal correlation to stay ahead of an evolving
   disturbance that static EFC re-chases each frame (M4).

That is the publishable claim: RL-learned focal-plane sensing that matches classical EFC in
the easy regime and beats it in the hard, realistic ones.
