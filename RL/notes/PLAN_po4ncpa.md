# Plan — replicate PO4NCPA (model-based RL) on our setup

**Paper.** Nousiainen, Taskin, Kasper, Orban de Xivry & Absil, *"Focal-plane
wavefront control with model-based reinforcement learning,"* A&A 709, A267 (2026)
/ arXiv:2604.00993. Algorithm: **PO4NCPA** (Policy Optimization for NCPAs).

**Why this paper.** It is the direct successor to Gutierrez et al. (2024), which we
reproduced — and it targets *exactly* our open problem: model-free PPO plateaus on
the coronagraphic dark hole because a dark image carries almost no exploration
signal (our "sensing limit"). PO4NCPA replaces blind model-free exploration with a
**learned, differentiable dynamics model** of the focal-plane image, so the policy
gets an analytic gradient of predicted focal-plane light w.r.t. the command even
where exploration collapses.

## The algorithm (as implemented in `src/po4ncpa.py`)

Two networks, trained jointly (Algorithm 1, one cycle per episode):

1. **Dynamics ensemble** `DynamicsNet` ×5 — a small U-Net that predicts the next
   preprocessed focal image `o_{t+1}` from `(o_t, o_{t-1}, a_{t-1}, a_t)`. Trained by
   supervised one-step prediction with the paper's **relative MSE** (Eq. 12).
2. **Policy** `PolicyNet` — maps the state `s_t = (o_t, o_{t-1}, a_{t-1})` to an
   absolute command `a_t ∈ [-1,1]^N`. Trained by **back-propagating the reward
   through H-step imagined rollouts** of the (frozen) dynamics ensemble, with the
   horizon `H` randomly sampled in `[2,7]` each update (Eq. 14).

Key design points faithful to the paper:
- **Sequential phase diversity:** state carries `(o_{t-1}, a_{t-1})`; the natural
  step-to-step command change is the implicit probe. No defocus/probe diversity
  (`diversity="none"`, single-channel science obs).
- **Cube-root residual preprocessing (Eq. 7):** `o = cbrt(PSF − PSF_ideal)`, signed.
  Reward `r = −‖o_{t+1}‖²` (total residual focal-plane light = dark-hole contrast
  under the coronagraph). No learned reward model — the reward is an analytic
  function of the predicted image.
- **Absolute action** scaled by the max expected NCPA per mode (`action_scale =
  max_abs_actuator`), so the normalized command equals the policy output.

## Backend reuse (no physics rewritten)

Runs on the existing `CoronagraphOptics` / `CoronagraphEnv`. New env hooks added
(all additive, backward-compatible):
- `image_scale="raw"` — raw normalized intensity (PO4NCPA does its own preprocessing).
- `diversity="none"` — single-channel science obs (the frame pair is the diversity).
- `env.ideal_image()` — cached unaberrated/flat reference PSF for the residual.

## Milestones (decided with user)

1. **Static NCPA, non-coronagraphic Strehl** — correctness check. Target: near-optimal
   Strehl (~0.99), matching our PPO replication and the paper's 99.4%.
   - smoke: `slurm/po4ncpa_sanity.slurm`; full: `slurm/po4ncpa_strehl.slurm` →
     `logs/po4ncpa_strehl/`.
2. **Static NCPA, coronagraph contrast** — the real test: break the ~3e-4 PPO
   plateau toward the expert/sim floor (1.285e-10). (`--use-coronagraph true`,
   IWA/OWA inside the FoV, optionally `--photon-flux`.)
3. **Dynamic NCPA** — add a temporal (frozen-flow / AR) Zernike disturbance to the
   env and reproduce the paper's headline long-exposure results.

## Config notes
- GPU: Adroit `gpu` partition (A100/V100); `dmrl2` has CUDA torch 2.8. The NN runs on
  GPU; hcipy propagation stays on CPU (4 threads). Optics image ≈ 33×33 (q=3,
  num_airy=5.5), 20 ideal Zernike modes (paper uses 55 — bump later).
