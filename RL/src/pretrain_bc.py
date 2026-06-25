"""Warm-start the PPO policy by behavior-cloning the least-squares expert.

Pipeline (see expert.py for the supervisory signal):
  1. Generate (observation -> modal action) pairs by driving the env with the
     phase-conjugation expert. Each step's target action is the *clipped
     increment* toward the full least-squares correction, so the dataset covers
     the whole closed-loop trajectory (large corrections early, near-zero late).
  2. Behavior-clone the PPO policy network: regress its Gaussian mean onto the
     expert action (MSE) over the dataset.
  3. Save the cloned model so train_ppo.py --init-model can RL-fine-tune from it.

Data generation is parallelized with a process pool because the expert reads the
optics' ground-truth aberration (only available in-process, not across a vec-env
boundary), so each worker owns its own env + expert.

All compute runs under SLURM on Adroit (conda env dmrl2); never on the login node.

Example:
  python pretrain_bc.py --num-episodes 800 --num-workers 8 --bc-epochs 30 \
      --num-control-modes 20 --out logs/bc_modal20/bc_policy
"""
from __future__ import annotations
import os

# Pin BLAS / FFT / numexpr to one thread per process so the data-generation pool
# scales with --num-workers instead of oversubscribing cores. Overridable.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import numpy as np

from environment import CoronagraphEnv
from expert import LeastSquaresExpert


# ---------------------------------------------------------------------------
# Data generation (runs in worker processes)
# ---------------------------------------------------------------------------
def _generate_shard(payload):
    """Generate expert trajectories for one worker. Returns (obs, act) arrays."""
    env_kwargs, n_eps, base_seed = payload
    env = CoronagraphEnv(**env_kwargs)
    expert = LeastSquaresExpert(env.optics, env.modal_basis)
    action_scale = env.action_scale
    max_steps = env.max_steps
    num_modes = int(env.num_control_modes)

    obs_list, act_list = [], []
    for e in range(n_eps):
        obs, _ = env.reset(seed=base_seed + e)
        c_target = expert.correction()              # full correction (meters)
        c_applied = np.zeros(num_modes)
        for _ in range(max_steps):
            remaining = c_target - c_applied
            action = np.clip(remaining / action_scale, -1.0, 1.0)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(action.astype(np.float32))
            obs, _, term, trunc, _ = env.step(action)
            c_applied = c_applied + action * action_scale
            if term or trunc:
                break
    return np.stack(obs_list), np.stack(act_list)


def generate_dataset(env_kwargs: dict, num_episodes: int, num_workers: int):
    """Parallel expert rollouts -> (obs[N,2,H,W], act[N,modes]) float32 arrays."""
    per = num_episodes // num_workers
    extra = num_episodes - per * num_workers
    payloads = []
    for w in range(num_workers):
        n_eps = per + (1 if w < extra else 0)
        if n_eps == 0:
            continue
        payloads.append((env_kwargs, n_eps, w * 100_000))  # disjoint seed ranges

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(payloads)) as pool:
        shards = pool.map(_generate_shard, payloads)
    obs = np.concatenate([s[0] for s in shards], axis=0)
    act = np.concatenate([s[1] for s in shards], axis=0)
    return obs, act


# ---------------------------------------------------------------------------
# Expert sanity check and policy evaluation (cheap, single process)
# ---------------------------------------------------------------------------
def _metric(env, info):
    """Episode metric for the env's objective: Strehl (higher better) or contrast."""
    return info["strehl"] if env.objective == "strehl" else info["contrast"]


def validate_expert(env_kwargs: dict, n: int, seed0: int = 5000) -> str:
    """One-shot full expert correction on fresh aberrations; return a summary line.

    Works for both the ideal Zernike corrector (applied via set_correction_modes)
    and the DM modal basis, and reports Strehl or contrast per the objective."""
    env = CoronagraphEnv(**env_kwargs)
    expert = LeastSquaresExpert(env.optics, env.modal_basis)
    strehl_mode = env.objective == "strehl"
    starts, finals = [], []
    for e in range(n):
        _, info = env.reset(seed=seed0 + e)
        starts.append(_metric(env, info))
        c = expert.correction()
        if env.ideal_modal_correction:
            env.optics.set_correction_modes(c)
        else:
            env.optics.set_actuators(env.modal_basis @ c)
        if strehl_mode:
            finals.append(float(np.max(env.optics.normalized_intensity(coronagraph=False, lyot=False))))
        else:
            finals.append(env.optics.dark_hole_contrast())
    s0, sf = float(np.mean(starts)), float(np.mean(finals))
    if strehl_mode:
        return f"start Strehl {s0:.4f} -> expert Strehl {sf:.4f}"
    return f"start contrast {s0:.3e} -> expert contrast {sf:.3e} ({s0 / max(sf, 1e-300):.1f}x)"


def eval_policy(model, env_kwargs: dict, n: int, seed0: int = 20000) -> str:
    """Deterministic rollout of a (cloned) policy; return a summary line."""
    env = CoronagraphEnv(**env_kwargs)
    strehl_mode = env.objective == "strehl"
    starts, finals, rets = [], [], []
    for e in range(n):
        obs, info = env.reset(seed=seed0 + e)
        starts.append(_metric(env, info)); ret = 0.0; done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += float(r); done = term or trunc
        finals.append(_metric(env, info)); rets.append(ret)
    s0, sf, r = float(np.mean(starts)), float(np.mean(finals)), float(np.mean(rets))
    if strehl_mode:
        return f"start Strehl {s0:.4f} -> final Strehl {sf:.4f}  return {r:+.3f}"
    return f"start {s0:.3e} -> final {sf:.3e}  return {r:+.3f} ({s0 / max(sf, 1e-300):.1f}x)"


# ---------------------------------------------------------------------------
# Behavior cloning
# ---------------------------------------------------------------------------
def behavior_clone(model, obs_np, act_np, epochs: int, batch: int, lr: float):
    """Regress the policy's Gaussian mean onto the expert action (MSE)."""
    import torch
    from torch.nn import functional as F

    policy = model.policy
    device = policy.device
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    obs_t = torch.as_tensor(obs_np, device=device)
    act_t = torch.as_tensor(act_np, device=device)
    n = obs_t.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            features = policy.extract_features(obs_t[idx])
            latent_pi, _ = policy.mlp_extractor(features)
            mean_actions = policy.action_net(latent_pi)
            loss = F.mse_loss(mean_actions, act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item(); nb += 1
        print(f"[bc] epoch {epoch + 1:3d}/{epochs}  mse={total / max(nb, 1):.6f}", flush=True)

    policy.set_training_mode(False)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Behavior-clone PPO policy from the LS expert")
    # env / optics (must match the fine-tune env)
    p.add_argument("--num-actuators-across", type=int, default=16)
    p.add_argument("--num-control-modes", type=int, default=20)
    p.add_argument("--pupil-pixels", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--action-scale", type=float, default=1e-8)
    p.add_argument("--rms-min", type=float, default=0.02)
    p.add_argument("--rms-max", type=float, default=0.08)
    # Config flags (must match the fine-tune / comparison env).
    p.add_argument("--ideal-modal-correction", action="store_true")
    p.add_argument("--num-aberration-modes", type=int, default=20)
    p.add_argument("--aberration-spectrum", choices=["white", "power_law"], default="white")
    p.add_argument("--psd-exponent", type=float, default=2.0)
    p.add_argument("--no-coronagraph", dest="use_coronagraph", action="store_false")
    p.add_argument("--objective", choices=["contrast", "strehl"], default="contrast")
    p.add_argument("--diversity", choices=["probe", "defocus"], default="probe")
    p.add_argument("--defocus-rad", type=float, default=1.0)
    p.add_argument("--flatten-obs", action="store_true")
    p.add_argument("--image-scale", choices=["log", "linear"], default="log")
    p.add_argument("--linear-ceil", type=float, default=1.0)
    p.add_argument("--net-arch", type=str, default=None)
    # data / BC
    p.add_argument("--num-episodes", type=int, default=800)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--bc-epochs", type=int, default=30)
    p.add_argument("--bc-batch", type=int, default=256)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="logs/bc_modal20/bc_policy")
    return p.parse_args()


def main():
    args = parse_args()
    if args.num_control_modes is None:
        raise SystemExit("pretrain_bc requires --num-control-modes (modal action space).")

    import torch
    torch.set_num_threads(max(1, args.num_workers))

    env_kwargs = dict(
        num_actuators_across=args.num_actuators_across,
        num_control_modes=args.num_control_modes,
        ideal_modal_correction=args.ideal_modal_correction,
        pupil_pixels=args.pupil_pixels,
        max_steps=args.max_steps,
        action_scale=args.action_scale,
        rms_min_waves=args.rms_min,
        rms_max_waves=args.rms_max,
        num_aberration_modes=args.num_aberration_modes,
        aberration_spectrum=args.aberration_spectrum,
        psd_exponent=args.psd_exponent,
        use_coronagraph=args.use_coronagraph,
        objective=args.objective,
        diversity=args.diversity,
        defocus_rad=args.defocus_rad,
        flatten_obs=args.flatten_obs,
        image_scale=args.image_scale,
        linear_ceil=args.linear_ceil,
    )

    # 0) Expert sanity check -- prove the supervisory signal is strong first.
    print(f"[expert] {validate_expert(env_kwargs, n=20)}", flush=True)

    # 1) Generate the imitation dataset in parallel.
    print(f"[data] generating {args.num_episodes} episodes on {args.num_workers} workers ...", flush=True)
    obs, act = generate_dataset(env_kwargs, args.num_episodes, args.num_workers)
    print(f"[data] {obs.shape[0]} pairs  obs={obs.shape}  act={act.shape}  "
          f"|action| mean={np.mean(np.abs(act)):.3f} sat={np.mean(np.abs(act) > 0.999):.2f}", flush=True)

    # 2) Build a PPO model identical to train_ppo and behavior-clone its policy.
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec = DummyVecEnv([lambda: CoronagraphEnv(**env_kwargs)])
    policy = "MlpPolicy" if args.flatten_obs else "CnnPolicy"
    policy_kwargs = dict(normalize_images=False, log_std_init=args.log_std_init)
    if args.net_arch:
        policy_kwargs["net_arch"] = [int(x) for x in args.net_arch.split(",") if x.strip()]
    model = PPO(
        policy=policy,
        env=vec,
        policy_kwargs=policy_kwargs,
        n_steps=max(64, args.max_steps * 8),   # buffer only used for construction; BC bypasses rollouts
        batch_size=256,
        target_kl=0.05,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        verbose=0,
        seed=args.seed,
    )

    # Baseline (random init) deterministic performance, for reference.
    print(f"[eval] pre-BC : {eval_policy(model, env_kwargs, n=args.eval_episodes)}", flush=True)

    behavior_clone(model, obs, act, epochs=args.bc_epochs, batch=args.bc_batch, lr=args.bc_lr)

    # 3) Evaluate the cloned policy and save.
    print(f"[eval] post-BC: {eval_policy(model, env_kwargs, n=args.eval_episodes)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"[save] cloned model -> {args.out}.zip", flush=True)


if __name__ == "__main__":
    main()
