# Plan: replicate Gutierrez et al. (2024), non-coronagraphic experiment

**Goal.** Reproduce the headline result of [Gutierrez et al. 2024, arXiv:2407.18733](https://arxiv.org/abs/2407.18733):
*pure PPO, no coronagraph, driving an ideal Zernike corrector to **Strehl ratio > 0.99**
from a λ/4-RMS aberration, observing only focal-plane phase-diversity images.*

This is the experiment the paper actually got "nearly perfect" results on. (Their
coronagraphic case only reached 2e-4 → 4e-5, ~5×.) Replicating the non-coronagraphic
case first validates our whole RL stack against a known-good target before we attempt
deep contrast.

We already proved the physics is correctable: with the ideal corrector, exact phase
conjugation reaches the 6.6e-11 contrast floor / Strehl ≈ 1.0 on every aberration. So
this is purely a **learning** replication, not a physics question.

---

## 1. Target configuration (paper → our code)

| Element | Paper (non-corono) | Our current default | Change needed |
|---|---|---|---|
| Coronagraph | **none** (direct PSF) | vortex + Lyot always on | add `use_coronagraph=False` path |
| Corrector | ideal Zernike, 21 modes | ideal corrector ✅ (`num_control_modes`) | set to **21** |
| Aberration | 21 Zernike modes, **1/f² PSD**, RMS **λ/4 (125 nm)** | 20 modes, white spectrum, RMS 0.02–0.08 λ | add power-law spectrum; RMS ≈ 0.25 λ; 21 modes |
| Observation | in-focus PSF + **defocused** PSF, flattened, **+ previous DM command** | 2 log images (science + actuator probe), no command | defocus diversity; **append command**; non-corono PSF |
| Action | 21 Zernike amplitudes | modal increment ✅ | keep (21-dim) |
| Reward | `r = −(1 − SR)^(2/5)` (absolute, per step) | `log10(prev/curr)` contrast | add `objective="strehl"` |
| Episode length | ~4 steps (10M steps / 2.5M episodes) | 20 | make tunable; try 4–8 |
| Training | **10M timesteps** | 200k | long run + checkpointing |
| Result | **Strehl > 0.99** | — | success criterion |

Decision notes:
- **Architecture.** Paper flattens images and concatenates the command into an MLP. We
  will use SB3 `MultiInputPolicy` with a `Dict` observation `{"image": (2,H,W),
  "command": (21,)}` — a CNN on the image plus an MLP on the command. Cleaner than
  manual flattening and keeps spatial structure. (Fallback: flatten + `MlpPolicy` for a
  more literal replication if MultiInput underperforms.)
- **1/f² spectrum.** PSD ∝ f⁻² ⇒ per-mode coefficient std ∝ 1/f. Use each Zernike's Noll
  radial order `n` as the spatial-frequency proxy (`f ∝ n`), so `std_i ∝ 1/(n_i)`. This
  concentrates variance in low-order modes, matching the paper (and making the problem
  easier than our current white spectrum).
- **Defocus diversity.** Second channel is the PSF with a known defocus (Zernike n=2,m=0)
  added for that exposure only. Defocus breaks the even-aberration sign ambiguity of a
  single in-focus PSF. Amplitude ~1 rad RMS, tunable.

---

## 2. Code changes (file by file)

### `optics.py`
1. `set_random_aberration(rms_waves, rng, spectrum="white", psd_exponent=2.0)`:
   when `spectrum="power_law"`, scale coefficient `i` by `1/(n_i)^(psd_exponent/2)`
   before summing, where `n_i` is the Noll radial order of mode `i` (`starting_mode=2`).
   Add a cached `_radial_orders` array. Keep `white` as the default so nothing else
   changes.
2. Add an optional `extra_phase=None` argument to `_focal_intensity` /
   `normalized_intensity`: a pupil phase added for that single exposure only (used for
   the defocus-diversity image). Restore afterwards (mirrors the existing
   `probe_actuators` save/restore).
3. Expose `num_aberration_modes` is already a constructor arg — just thread it through the
   env. Confirm aberration basis count == corrector mode count == 21 for exact
   correctability.
4. (Optimization) In non-corono mode the science PSF and `strehl()` share the same bare
   propagation — compute the bare PSF once per step and derive both.

### `environment.py`
1. New constructor args:
   - `use_coronagraph: bool = True` — when False, the science channel is the bare PSF
     (`coronagraph=False, lyot=False`) and the episode metric is Strehl.
   - `objective: str = "contrast"` — `"strehl"` switches the reward to `−(1−SR)^(2/5)`.
   - `diversity: str = "probe"` — `"defocus"` builds channel 1 from a known defocus phase.
   - `defocus_rad: float = 1.0` — defocus diversity amplitude.
   - `aberration_spectrum: str = "white"`, `psd_exponent: float = 2.0`,
     `num_aberration_modes: int = 21`.
   - `include_command: bool = False` — when True, observation becomes a `Dict`
     `{"image", "command"}` and the previous (normalized) corrector coeffs are appended.
2. `reset`: store `self._prev_command = zeros(n_modes)`; clear corrector.
3. `_observe`: build channel 0 from the bare PSF when `use_coronagraph=False`; build
   channel 1 via defocus when `diversity="defocus"`. Return Strehl alongside contrast so
   the reward can use either. Assemble the `Dict` obs when `include_command`.
4. `step`: compute reward from `objective`. Update `_prev_command` from
   `optics.correction_coeffs / max_abs_actuator` (normalized to ~[-1,1]).
5. Tune image normalization for the PSF: the bare PSF peak is ~Strehl (≫ contrast), so the
   current `log_floor=-12, log_ceil=0` log scaling wastes range. Add `log_floor`/`log_ceil`
   defaults appropriate for PSF images, or a `linear` scaling option. Validate the obs
   spans a useful [0,1] range in the sanity job.

### `train_ppo.py`
1. New flags: `--use-coronagraph/--no-coronagraph`, `--objective {contrast,strehl}`,
   `--diversity {probe,defocus}`, `--defocus-rad`, `--num-aberration-modes`,
   `--aberration-spectrum {white,power_law}`, `--psd-exponent`, `--include-command`.
2. Auto-select policy: `MultiInputPolicy` when the env obs is a `Dict`, else `CnnPolicy`.
3. `ContrastCallback`: also log `env/strehl` as the primary metric; log
   `eval/mean_reward` translation note in TB.
4. Add a `CheckpointCallback` (e.g. every 500k steps) so a 10M-step run is resumable and
   never loses progress; support `--resume <checkpoint>` to continue.

### New SLURM scripts
- `replicate_sanity.slurm` (~15 min): env builds, `check_env` passes on the `Dict` obs,
  expert one-shot → **Strehl ≈ 1.0**, obs range sane, 10k-step PPO smoke shows reward
  rising. Includes the mail directives.
- `replicate_gutierrez.slurm`: the full run (see protocol). Positional args for steps /
  run name. Mail directives included.

---

## 3. Training protocol (phased — don't jump straight to 10M)

**Phase 0 — sanity (~15 min).** `replicate_sanity.slurm`. Gate: expert reaches Strehl
≈ 1.0 through the corrector; PPO smoke reward trends positive; obs in a healthy [0,1].

**Phase 1 — tuning run (~1–2 h, 500k–1M steps, 16 envs).** Confirm non-corono throughput
(should be ≫ the coronagraphic ~20 fps since the expensive `vortex_q=256` propagation is
gone), pick episode length (try 4 and 8), `learning_rate` (3e-4), `log_std_init` (−1),
and image normalization. Gate: Strehl clearly climbing past ~0.9.

**Phase 2 — full replication (10M steps, checkpointed).** Match the paper's budget. Run
as one long job or a chain of resumable sbatch jobs (checkpoint every 500k). Track
deterministic eval Strehl vs. timesteps and overlay the paper's curve.

---

## 4. Success criteria
- **Primary:** deterministic eval **mean Strehl > 0.99** from λ/4-RMS, 1/f², 21-mode
  aberrations, within a short (≤ ~8-step) episode — matching the paper.
- **Secondary:** learning-curve shape comparable to the paper (most gains within a few M
  steps); robustness across the aberration distribution (low eval variance).

## 5. Risks & open decisions
- **MultiInput vs flatten+MLP** — start MultiInput; fall back to literal flatten+MLP if it
  stalls.
- **Exact 1/f² mapping** — radial-order proxy is an approximation; if results diverge from
  the paper, refine the per-mode weighting to a true cycles-per-aperture frequency.
- **PSF image normalization** — most likely tuning point; verify in Phase 0.
- **Episode length** — paper implies ~4 steps; our 20-step default may slow learning.
- **10M-step wall time** — unknown until Phase 1 measures non-corono throughput; mitigate
  with checkpointing and a resumable job chain.
- **Reward variant** — if `−(1−SR)^(2/5)` is slow, a `log(1−SR)` shaping is a documented
  alternative.

## 6. Out of scope (deliberately deferred)
Coronagraphic deep contrast / 1e-10. This plan only validates the RL stack against the
paper's non-coronagraphic Strehl result. Once Strehl > 0.99 is reproduced, the corrected
wavefront fed through the existing vortex+Lyot is the natural bridge back to the contrast
goal.
