# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Operational constraint (read first)

This repo runs on **Princeton's Adroit cluster**. **Do not run simulations, training, or any compute-heavy Python on the login node** — always submit via `sbatch`. The conda env is `dmrl2` at
`/scratch/network/ak9088/anaconda3/envs/dmrl2` and is activated by every SLURM script.

What *is* allowed on the login node:
- `sbatch`, `squeue`, `sacct`, file reads, `git`, `ls`, `grep`.
- `python -m py_compile <file>` for syntax checks (no imports of hcipy / numpy heavy work happens).

What is **not** allowed: running `python optics.py`, `python environment.py`, `python train_ppo.py`, or any script that builds a `CoronagraphOptics` / propagates wavefronts. Those go through SLURM even for "quick" tests — the project's `rl_sanity.slurm` is the smallest viable smoke test.

Waiting on a job: prefer a background `bash` that polls `squeue` until the job leaves the queue, then `grep`s the slurm output. The harness notifies on its completion. Don't loop with short sleeps.

**Always email the user on job completion.** Every SLURM script — existing or new — must include these two directives in its `#SBATCH` header so the user is notified when a job ends or fails:

```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak9088@princeton.edu
```

Use the Princeton address `ak9088@princeton.edu`: Adroit's mail relay only reliably delivers to `@princeton.edu`, so an external gmail target is silently dropped. When writing any new `.slurm` script, add these two directives by default.

## Live code lives in `RL/`

Everything outside `RL/` is legacy: `Baselines/`, `Deprecated/`, the top-level notebooks and PNGs, and `RL/legacy/` (43 archived items — old `archs/`, `data/`, `evals/`, `jobs/`, `logs/`, `models/`, old `*.py`/`*.ipynb`/`*.slurm`). The active project is six Python modules and their SLURM scripts in `RL/`. Don't restore from `legacy/` without a reason.

**Directory layout (reorganized 2026-06-21).** `RL/` is split into subdirectories:

```
RL/
├── src/        python modules: optics, environment, expert, pretrain_bc, train_ppo, efc_floor
├── slurm/      all .slurm batch scripts
├── notes/      markdown docs (PROJECT_STATUS.md, RESULTS_*, PLAN_*)
├── outputs/    slurm-<jobid>.out console logs
├── logs/       training runs (tensorboard + models/best_model.zip + checkpoints)
└── legacy/     archived earlier work
```

**Submit jobs from the `RL/` root**: `sbatch slurm/<script>.slurm [args]`. Each script
`cd`s to the repo root, exports `PYTHONPATH=…/RL/src` (so both `python src/<module>.py`
and the inline `python - <<PY` eval heredocs import the package), and writes its console
log to `outputs/slurm-%j.out`. The high-level status lives in
[`RL/notes/PROJECT_STATUS.md`](RL/notes/PROJECT_STATUS.md).

## Architecture: the pipeline, end-to-end

The project is a single linear pipeline. Each stage has one Python entry point and one SLURM script.

```
optics.py  →  environment.py  →  expert.py  →  pretrain_bc.py  →  train_ppo.py
   (physics)    (RL env)         (LS supervisor)  (BC warm-start)   (PPO fine-tune)

                         efc_floor.py  (standalone diagnostic, not in the pipeline)
```

**`optics.py` — `CoronagraphOptics`.** Single deformable mirror in the pupil ahead of a charge-2 vortex + Lyot stop, propagated to a science focal plane via `hcipy`. Owns the disturbance model (`set_random_aberration` draws low-order Zernike phase screens from `_aberration_basis`, scaled to a given RMS in waves) and the DM (`hcipy.DeformableMirror` with Gaussian influence functions). `dark_hole_contrast` is mean normalized intensity inside the 3–12 λ/D annulus; the unaberrated floor is `~6.6e-11`. `build_modal_actuator_basis(num_modes)` returns actuator commands (n_act × n_modes) that LS-fit each Zernike onto the DM surface, normalized to unit surface-RMS per mode — this is what bridges modal control to a real actuator grid.

**`environment.py` — `CoronagraphEnv`.** Gymnasium env. Observation is a `(2, H, W)` log-scaled normalized-intensity image (channel 0 = science, channel 1 = a fixed seeded phase-diversity probe), `float32` in `[0, 1]`. With defaults (`q=2`, `num_airy=14`), `H=W=56`. Action is a `Box([-1,1], shape=(action_dim,))` *increment* per step, scaled by `action_scale` (meters of surface). The DM accumulates over the 20-step episode. If `num_control_modes` is set, action lives in modal space and is mapped to actuator deltas via `modal_basis`. **Reward is the log-ratio `log10(prev_contrast / contrast)`** — decades dug per step; the episode return telescopes to `log10(c_start / c_final)`, so a return of `+1.0` means a 10× contrast improvement.

**`expert.py` — `LeastSquaresExpert`.** Phase-conjugation supervisor for the modal env. Precomputes the per-mode pupil-phase response `P[:,i] = 2k · surface(B[:,i])` over the aperture (the factor of 2 is hcipy's reflection OPD: `DeformableMirror.opd = 2·surface`), then `correction() = pinv(P) @ (-aberration_phase)`. Reads ground-truth `optics.aberration_phase` — works only in-process, which is why data generation must own its own env per worker.

**`pretrain_bc.py` — behavior cloning.** Three steps in one script: (0) sanity-check the expert by one-shot applying its correction to N aberrations and printing the improvement; (1) parallel data generation — each worker owns its own env+expert and rolls out the *closed-loop* expert, where the target action at each step is `clip((c_target − c_applied) / action_scale, −1, 1)`, so the dataset spans the whole trajectory (saturated `±1` early, near-zero late); (2) BC via hand-rolled MSE on the SB3 policy's `extract_features → mlp_extractor → action_net` chain. Then deterministic eval, then `model.save()`. Multiprocessing uses the `spawn` start method.

**`train_ppo.py` — PPO + warm-start.** `CnnPolicy`, `normalize_images=False` (obs already in `[0,1]`), `n_steps=max_steps*8`, `batch_size=256`, `target_kl=0.05`. Logs `env/contrast`, `env/log10_contrast`, `env/strehl` via `ContrastCallback`. The `--init-model PATH` flag loads donor PPO weights into the fresh model's policy via `load_state_dict`, then **re-applies `--log-std-init` to `policy.log_std`** — without that, the donor's saved `log_std` would override the CLI and exploration noise (≈0.37) would swamp the tiny cloned actions (mean `|a|≈0.04`).

**`efc_floor.py` — standalone diagnostic.** Perfect-knowledge electric-field conjugation: builds the complex dark-hole Jacobian by poking each actuator, SVDs it, then iteratively solves the regularized LS command (line-searched over rcond) that drives the dark-hole field to zero. Run for 16/24/32 DMs to measure the achievable contrast floor independent of any learned controller.

### Critical sign / scale conventions

- DM phase added to the wavefront is `2·k·surface` (factor 2 for reflection — see hcipy's `DeformableMirror.opd`). Phase-conjugation correction is therefore `surface_target = −aberration_phase / (2k)`.
- Modal basis is normalized to **unit surface RMS** per mode, so a modal coefficient is in meters of surface RMS, and `action × action_scale` is the per-step modal increment in the same units. `action_scale=1e-8` over 20 max-saturated steps gives 2e-7 m of cumulative modal stroke per mode — comfortably above the ~6e-8 m needed for the default 0.02–0.08 wave aberration range.
- The aberration basis and the modal control basis use the **same** `make_zernike_basis(..., starting_mode=2)` and (by default) the same number of modes, so for a fixed `pupil_pixels` the same RNG seed produces the same disturbance regardless of `num_actuators_across` — fair comparisons across DM sizes.

## Running things

### SLURM scripts (paired with each entry point)

| Script | What it runs | Typical wall time |
|---|---|---|
| `rl_sanity.slurm` | `optics.py` floor + `check_env` + 4k-step PPO smoke test | ~5 min |
| `rl_test.slurm` | short PPO (50k steps) with modal-20 action space | ~30 min |
| `rl_job.slurm` | full PPO (1M steps, raw actuators) | hours |
| `pretrain.slurm` | BC: expert validation → parallel data gen → BC → eval → save | ~5 min |
| `finetune.slurm` | `train_ppo.py --init-model logs/bc_modal20/bc_policy` | ~25 min for 100k steps |
| `efc_floor.slurm` | floor diagnostic across DM sizes | ~3 min |

Submit from the `RL/` root: `sbatch slurm/<script>.slurm`. Output: `outputs/slurm-<jobid>.out`. Best checkpoints are saved under `logs/<run_name>/models/best_model.zip` by `EvalCallback`; final models go to `logs/<run_name>/<tb_log_name>_final.zip`.

### Threading

Every SLURM script and Python entry exports / `setdefault`s `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=NUMEXPR_MAX_THREADS=1` so that throughput scales with `--num-envs` (PPO) or `--num-workers` (BC pool) instead of having each process oversubscribe cores. `efc_floor.slurm` is the exception — it's single-process and sets these to 4 so SVD + FFT can use multiple threads.

### TensorBoard

`python -m tensorboard --logdir RL/logs`. Note that `env/log10_contrast` is the *mean over all rollout steps* (includes the high-contrast start of every episode and stochastic exploration), so it understates per-episode end performance. The deterministic eval return from `EvalCallback` is the cleaner metric: it's `log10(c_start / c_final)` averaged over the eval episodes.

## Current project state

**Latest milestone (2026-06-19): reproduced [Gutierrez et al. 2024](https://arxiv.org/abs/2407.18733) non-coronagraphic result — pure PPO reaches Strehl 0.988 (median 0.999, 90% > 0.99).** Full config, numbers, and the exact command are in [`RL/notes/RESULTS_paper_replication.md`](RL/notes/RESULTS_paper_replication.md); the run is `replicate_paper.slurm` → `logs/paper_replication/`. The decisive settings that made model-free PPO finally learn: **γ=0** (greedy one-shot correction), **16×16 images** (533-dim input, not 56×56), **4-step episodes with the previous command in the observation**, and **absolute action** (`action_mode=absolute`). The env now supports these via flags on `CoronagraphEnv` (`use_coronagraph`, `objective`, `diversity`, `flatten_obs`, `image_scale`, `include_command`, `action_mode`, `aberration_spectrum`) and `train_ppo.py` (`--gamma`, `--lr-linear-decay`, `--batch-size`, `--target-kl`, `--q`, `--num-airy`, `--n-steps`). Remaining fidelity step: add SNR≈100 detector noise. The ideal Zernike corrector (`ideal_modal_correction`) is implemented on `CoronagraphOptics`.

**Coronagraphic follow-up (2026-06-20): the same recipe with the coronagraph ON, sensing from the dark image, plateaus at ~3e-4 — a sensing limit.** Full write-up in [`RL/notes/RESULTS_coronagraph.md`](RL/notes/RESULTS_coronagraph.md); run `replicate_corona.slurm` → `logs/corona_replication/`, eval `eval_corona.slurm`. With `objective=log_contrast` (reward `−log10(contrast)/10`), a small dark hole (IWA 2–OWA 4, fits the 16×16 FoV), and log image scaling `[-10,-2]`, PPO learns ~20× (6.7e-3 → 3.1e-4) then freezes at 600k steps. The deterministic policy corrects fully in step 1 and **does not refine on steps 2–4** (contrast identical to 4 sig figs): once the hole is at ~3e-4 the residual wavefront leaves no extractable signal in the dark image. The 21 modes *can* reach 1.28e-10 (the expert does by flattening), so this is an information/sensing limit, not control — reproducing the paper's coronagraphic wall (~5×). Untested next step: a **"both" observation** (bright PSF for sensing + coronagraphic image for nulling) to break the plateau.

**Contrast-as-reward is degenerate (2026-06-21): killed.** Rewarding the agent directly on dark-hole contrast (instead of Strehl) collapses to **PSF wrecking**. Non-coronagraphically (3 reward shapes: `-log10(C)/10`, `-C/ceil`, `log10(C_prev/C)`) all three null the 2–4 λ/D annulus *below the diffraction floor* by smearing the Airy core — Strehl 0.34 → 0.003. Turning the coronagraph back on and starting from near-perfect wavefronts did **not** help: in a low-aberration sweep (rms 0.02/0.05/0.10 waves) the policy collapses to an **observation-independent constant command** that lands all three at ~2.4e-4 / Strehl ~0.002 — for rms 0.02 it makes an already-excellent Strehl-0.984 / 6e-5 wavefront *worse*. Diagnosis: weak dark-image sensing (the plateau above) means the policy can't learn the input→correction map, so it defaults to the easiest constant output, which for a contrast-*mean* reward is a moderate scattered hole; Strehl works only because the PSF peak is a strong always-available signal. Full write-up + tables in [`RL/notes/PROJECT_STATUS.md`](RL/notes/PROJECT_STATUS.md). Runs `logs/creward_{logc,negc,diffc}` and `logs/corona_lowab_rms{02,05,10}`; scripts `slurm/contrast_reward.slurm`, `slurm/corona_lowab.slurm` (+ matching `eval_*`). **Paths forward:** warm-start contrast fine-tune from BC/expert (keeps the policy in the refine basin), the "both" observation, or a throughput-protecting composite reward `-log10(C) - λ(1-Strehl)`.

### Earlier coronagraphic work (warm-start spine)
The model-free RL loop alone never learned to dig the dark hole (4 runs, eval return stuck near 0). The earlier working approach was **warm-start from a least-squares expert**: BC reached 7.4× contrast deterministically, and a 100k-step PPO fine-tune (`finetune.slurm`) pushed it to 12.9× (eval return +1.11) — surpassing the 8.1× phase-conjugation expert.

The `efc_floor.py` diagnostic then pinned the hard limit: with the 16×16 DM and a 12 λ/D OWA, the **controllable spatial-frequency band (N/2 λ/D = 8 λ/D) is narrower than the dark hole**, so no controller can do better than ~−7.3 log10 contrast on this geometry. To reach the stated −10 target requires either (a) a larger DM (24×24 reaches −8.7, 32×32 reaches −9.8), (b) a smaller OWA, or (c) replacing the influence-function DM with an *ideal Zernike corrector* (a paper-style idealized model where `N_act = N_modes` and one unit of mode i produces exactly Zernike i — this would make the existing phase-conjugation expert *exact*).

When extending the code, the warm-start pattern (expert → BC → `--init-model` fine-tune) is the spine to keep. The `ideal_modal_correction` path, if added, belongs as a flag on `CoronagraphOptics` that adds a `correction_phase` to `_wavefront()` and bypasses the DM, with `environment.py` routing the modal action through it instead of `dm.add_actuators`.
