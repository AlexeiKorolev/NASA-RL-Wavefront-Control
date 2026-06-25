"""Train a PPO agent to dig a dark hole on the CoronagraphEnv.

The observation is a 2-channel normalized-intensity image (science + phase
diversity), so we use a CNN policy. Images are already in [0, 1], hence
normalize_images=False.

Example:
  python train_ppo.py --total-timesteps 200000 --num-envs 4 --tb-log-name run1
TensorBoard:
  python -m tensorboard --logdir logs
"""
from __future__ import annotations
import os

# Pin BLAS/FFT threads before numpy is imported so each (sub)process stays
# single-threaded; throughput then scales with --num-envs instead of having
# parallel workers oversubscribe the cores. Overridable from the environment.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback

from environment import CoronagraphEnv


class ContrastCallback(BaseCallback):
    """Log mean dark-hole contrast / Strehl from step infos to TensorBoard."""
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        # Drop NaN contrasts (non-coronagraphic mode reports contrast as NaN).
        contrasts = [i["contrast"] for i in infos
                     if isinstance(i, dict) and "contrast" in i and np.isfinite(i["contrast"])]
        strehls = [i["strehl"] for i in infos if isinstance(i, dict) and "strehl" in i]
        if contrasts:
            self.model.logger.record("env/contrast", float(np.mean(contrasts)))
            self.model.logger.record("env/log10_contrast", float(np.mean(np.log10(contrasts))))
        if strehls:
            self.model.logger.record("env/strehl", float(np.mean(strehls)))
        return True


def make_env(env_kwargs: dict):
    def _thunk():
        return Monitor(CoronagraphEnv(**env_kwargs))
    return _thunk


def build_vec_env(n_envs: int, env_kwargs: dict):
    thunks = [make_env(env_kwargs) for _ in range(n_envs)]
    return SubprocVecEnv(thunks) if n_envs > 1 else DummyVecEnv(thunks)


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on CoronagraphEnv")
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-actuators-across", type=int, default=16)
    p.add_argument("--num-control-modes", type=int, default=None)
    p.add_argument("--ideal-modal-correction", action="store_true",
                   help="Bypass the DM and act directly on ideal Zernike correction "
                        "modes (requires --num-control-modes). Removes the DM "
                        "spatial-frequency bandwidth limit.")
    p.add_argument("--pupil-pixels", type=int, default=256)
    p.add_argument("--q", type=int, default=2, help="focal-plane samples per lambda/D")
    p.add_argument("--num-airy", type=float, default=14.0, help="focal half-extent in lambda/D (sets image size)")
    p.add_argument("--action-mode", choices=["incremental", "absolute"], default="incremental")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=None,
                   help="PPO rollout length per env (default max_steps*8). Set explicitly "
                        "for single-step episodes, where max_steps*8 is too small.")
    p.add_argument("--action-scale", type=float, default=1e-8)
    p.add_argument("--rms-min", type=float, default=0.02)
    p.add_argument("--rms-max", type=float, default=0.08)
    # Gutierrez-replication knobs (non-coronagraphic Strehl experiment).
    p.add_argument("--num-aberration-modes", type=int, default=20)
    p.add_argument("--aberration-spectrum", choices=["white", "power_law"], default="white")
    p.add_argument("--psd-exponent", type=float, default=2.0)
    p.add_argument("--no-coronagraph", dest="use_coronagraph", action="store_false",
                   help="Observe the bare PSF instead of the coronagraphic image.")
    p.add_argument("--objective", choices=["contrast", "strehl", "log_contrast", "neg_contrast"],
                   default="contrast")
    p.add_argument("--dark-hole-iwa", type=float, default=3.0)
    p.add_argument("--dark-hole-owa", type=float, default=12.0)
    p.add_argument("--log-floor", type=float, default=-12.0, help="log10 image-scaling floor")
    p.add_argument("--log-ceil", type=float, default=0.0, help="log10 image-scaling ceiling")
    p.add_argument("--diversity", choices=["probe", "defocus"], default="probe")
    p.add_argument("--defocus-rad", type=float, default=1.0)
    p.add_argument("--include-command", action="store_true",
                   help="Append the previous corrector command to the observation "
                        "(Dict obs + MultiInputPolicy); requires --ideal-modal-correction.")
    p.add_argument("--flatten-obs", action="store_true",
                   help="Flatten image(+command) into a vector and use MlpPolicy "
                        "(matches the paper; fixes the CNN critic that failed to learn).")
    p.add_argument("--image-scale", choices=["log", "linear"], default="log")
    p.add_argument("--linear-ceil", type=float, default=1.0)
    p.add_argument("--net-arch", type=str, default=None,
                   help="Comma-separated hidden sizes for the MLP, e.g. 256,256.")
    p.add_argument("--checkpoint-freq", type=int, default=0,
                   help="Save a resumable checkpoint every N steps (0 disables).")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-linear-decay", action="store_true",
                   help="Linearly decay the learning rate to 0 over training (paper setting).")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor. Paper uses 0 (greedy one-step correction).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--target-kl", type=float, default=0.05,
                   help="PPO target_kl early-stop. <=0 disables (SB3 default / paper setting).")
    p.add_argument("--ent-coef", type=float, default=0.0)
    # Initial log std of the Gaussian policy. SB3 default 0.0 (std=1) is too broad
    # for the 256-dim actuator action; lower values sharpen exploration.
    p.add_argument("--log-std-init", type=float, default=0.0)
    p.add_argument("--log-dir", type=str, default=os.path.join(os.path.dirname(__file__), "logs"))
    p.add_argument("--tb-log-name", type=str, default="ppo_coronagraph")
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=None)
    # Warm start: load behavior-cloned policy weights (see pretrain_bc.py) into a
    # fresh PPO so RL fine-tunes from the least-squares imitation, not random init.
    p.add_argument("--init-model", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    env_kwargs = dict(
        num_actuators_across=args.num_actuators_across,
        num_control_modes=args.num_control_modes,
        ideal_modal_correction=args.ideal_modal_correction,
        action_mode=args.action_mode,
        pupil_pixels=args.pupil_pixels,
        q=args.q,
        num_airy=args.num_airy,
        dark_hole_iwa=args.dark_hole_iwa,
        dark_hole_owa=args.dark_hole_owa,
        log_floor=args.log_floor,
        log_ceil=args.log_ceil,
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
        include_command=args.include_command,
        flatten_obs=args.flatten_obs,
        image_scale=args.image_scale,
        linear_ceil=args.linear_ceil,
    )

    env = build_vec_env(args.num_envs, env_kwargs)
    eval_env = build_vec_env(1, env_kwargs)

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.log_dir, "models"),
            log_path=args.log_dir,
            eval_freq=max(1, args.eval_freq // max(1, args.num_envs)),
            deterministic=True,
            render=False,
        ),
        ContrastCallback(),
    ]
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // max(1, args.num_envs)),
            save_path=os.path.join(args.log_dir, "checkpoints"),
            name_prefix=args.tb_log_name))

    # Policy: flat vector -> MlpPolicy; Dict (image+command) -> MultiInputPolicy; else CNN.
    if args.flatten_obs:
        policy = "MlpPolicy"
    elif args.include_command:
        policy = "MultiInputPolicy"
    else:
        policy = "CnnPolicy"

    policy_kwargs = dict(normalize_images=False, log_std_init=args.log_std_init)
    if args.net_arch:
        policy_kwargs["net_arch"] = [int(x) for x in args.net_arch.split(",") if x.strip()]

    # Learning rate: constant, or linearly decayed to 0 over training (paper setting).
    if args.lr_linear_decay:
        learning_rate = lambda progress_remaining, lr0=args.learning_rate: progress_remaining * lr0
    else:
        learning_rate = args.learning_rate

    model = PPO(
        policy=policy,
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=args.n_steps if args.n_steps else args.max_steps * 8,
        batch_size=args.batch_size,
        gamma=args.gamma,
        # target_kl early-stops runaway epochs; <=0 disables it (SB3 default / paper).
        target_kl=args.target_kl if args.target_kl and args.target_kl > 0 else None,
        learning_rate=learning_rate,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
    )

    if args.init_model:
        # Copy the behavior-cloned weights into this PPO's policy (same arch),
        # keeping PPO's fresh optimizer / rollout buffer for fine-tuning.
        import torch
        donor = PPO.load(args.init_model, device=model.device)
        model.policy.load_state_dict(donor.policy.state_dict())
        # load_state_dict also copied the donor's log_std; re-apply --log-std-init
        # so fine-tune exploration is set here (the cloned actions are tiny, so an
        # inherited std=0.37 would swamp the warm-started policy).
        with torch.no_grad():
            model.policy.log_std.fill_(float(args.log_std_init))
        print(f"Warm-started policy from {args.init_model}; log_std set to {args.log_std_init}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=args.tb_log_name,
        progress_bar=True,
    )

    model_path = os.path.join(args.log_dir, f"{args.tb_log_name}_final")
    model.save(model_path)
    print(f"Saved final model to: {model_path}.zip")


if __name__ == "__main__":
    main()
