"""
Train a PPO agent on the CoronagraphEnvironment with safety guards to prevent
"observation doesn't match space" failures during training.

Features
- SafeCoronagraphEnvironment: overrides _get_obs to sanitize outputs and avoid assertions
- SafeObsWrapper: clips/cleans observations to finite bounds, enforces dtypes
- TensorBoard logging and evaluation callback
- Simple CLI for common hyperparameters

Usage (Windows PowerShell):
  python "A:\\Projects\\DM + RL\\RL\\train_ppo.py" --total-timesteps 200000 --tb-log-name ppo_run1

TensorBoard:
  python -m tensorboard --logdir "A:\\Projects\\DM + RL\\RL\\logs" --host localhost --port 6006
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env

# Local import
from environment import CoronagraphEnvironment


class SafeCoronagraphEnvironment(CoronagraphEnvironment):
    """Subclass that sanitizes observations and removes strict asserts.

    We override _get_obs() to:
    - Cast to float32
    - Replace NaNs/Infs
    - Keep shapes consistent
    - Do NOT assert against observation_space (wrapper will clip if needed)
    """
    def _get_obs(self):
        # Delegate to base implementation to preserve shapes/keys, then sanitize
        obs = super()._get_obs()
        for k, v in list(obs.items()):
            arr = np.asarray(v, dtype=np.float32)
            obs[k] = np.nan_to_num(arr, posinf=np.finfo(np.float32).max, neginf=0.0)
        return obs


class SafeObsWrapper(gym.ObservationWrapper):
    """Sanitize observations: dtype, NaNs/Infs, clip to finite bounds, enforce shapes."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.obs_space: gym.spaces.Dict = env.observation_space

    def observation(self, obs):
        def _fix(arr: np.ndarray, space: gym.spaces.Box):
            arr = np.asarray(arr, dtype=np.float32)
            arr = np.nan_to_num(arr, posinf=np.finfo(np.float32).max, neginf=0.0)
            # Clip only where bounds are finite
            low = np.where(np.isfinite(space.low), space.low, -np.inf).astype(np.float32)
            high = np.where(np.isfinite(space.high), space.high, np.inf).astype(np.float32)
            return np.clip(arr, low, high)
        obs["image"] = _fix(obs["image"], self.obs_space.spaces["image"])
        if "slopes" in obs and "slopes" in self.obs_space.spaces:
            obs["slopes"] = _fix(obs["slopes"], self.obs_space.spaces["slopes"])
        obs["contrast"] = _fix(obs["contrast"], self.obs_space.spaces["contrast"])
        return obs


class TBContrastCallback(BaseCallback):
    """Example TensorBoard callback; logs custom keys if present in infos."""
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            cs = [i.get("contrast") for i in infos if isinstance(i, dict) and "contrast" in i]
            if cs:
                self.model.logger.record("train/contrast", float(np.mean(cs)))
        return True


def build_env(num_modes: int, pixels: int, oversizing_factor: float, num_airy: int, ppsr: int, include_slopes: bool = True, basis: str = "harmonic") -> gym.Env:
    env = SafeCoronagraphEnvironment(
        num_modes=num_modes,
        pixels=pixels,
        oversizing_factor=oversizing_factor,
        num_airy=num_airy,
        pixels_per_spacial_res=ppsr,
        coronagraph_charge=6,
        include_slopes=include_slopes,
        basis=basis
    )
    env = Monitor(env)
    # env = SafeObsWrapper(env)
    return env


def make_vec_env(n_envs: int, **env_kwargs) -> DummyVecEnv:
    def _thunk():
        return build_env(**env_kwargs)
    return DummyVecEnv([_thunk for _ in range(n_envs)])


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on CoronagraphEnvironment")
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--num-modes", type=int, default=4)
    p.add_argument("--basis", type=str, default="orthogonal")
    p.add_argument("--pixels", type=int, default=64)
    p.add_argument("--oversizing-factor", type=float, default=1.0)
    p.add_argument("--num-airy", type=int, default=5)
    p.add_argument("--ppsr", type=int, default=2, help="pixels_per_spacial_res")
    p.add_argument("--log-dir", type=str, default=os.path.join(os.path.dirname(__file__), "logs"))
    p.add_argument("--tb-log-name", type=str, default="ppo_coronagraph")
    p.add_argument("--eval-freq", type=int, default=5000)
    p.add_argument("--seed", type=int, default=None)
    # Slopes toggle
    p.add_argument("--include-slopes", dest="include_slopes", action="store_true", help="include slopes in observations (default)")
    p.add_argument("--no-slopes", dest="include_slopes", action="store_false", help="exclude slopes from observations")
    p.set_defaults(include_slopes=True)
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # One-off check on a single env (wrapped)
    try:
        check_env(build_env(
            num_modes=args.num_modes,
            pixels=args.pixels,
            oversizing_factor=args.oversizing_factor,
            num_airy=args.num_airy,
            ppsr=args.ppsr,
        ), warn=True)
    except Exception as exc:
        print("[WARN] Env check raised an exception (continuing):", exc)

    # Vectorized training and eval envs
    env_kwargs = dict(
        num_modes=args.num_modes,
        pixels=args.pixels,
        oversizing_factor=args.oversizing_factor,
        num_airy=args.num_airy,
        ppsr=args.ppsr,
        include_slopes=args.include_slopes,
        basis=args.basis,
    )
    env = make_vec_env(args.num_envs, **env_kwargs)
    eval_env = make_vec_env(1, **env_kwargs)

    # Logger and callbacks
    new_logger = configure(args.log_dir, ["stdout", "csv", "tensorboard"])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, "models"),
        log_path=args.log_dir,
        eval_freq=max(1, args.eval_freq),
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
    )
    model.set_logger(new_logger)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, TBContrastCallback()],
        tb_log_name=args.tb_log_name,
        progress_bar=True,
    )

    model_path = os.path.join(args.log_dir, f"{args.tb_log_name}_final")
    model.save(model_path)
    print(f"Saved final model to: {model_path}.zip")

    # Quick rollout sanity check
    test_env = build_env(**env_kwargs)
    obs, _ = test_env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = bool(terminated) or bool(truncated)
        steps += 1
    print("Rollout complete. Steps:", steps)


if __name__ == "__main__":
    main()
