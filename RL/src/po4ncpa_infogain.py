"""PO4NCPA + learned active probing with an OBSERVABLE-ONLY info-gain probe reward
(M2-honest -- notes/PLAN_active_probing.md).

Motivation. The reconstruction-auxiliary variant (po4ncpa_probe_recon.py) supervised a
FieldHead against the *true* complex dark-hole field. That field is unmeasurable on a real
space telescope (the detector only ever records intensity), so it is off-goal for a
controller meant to be trainable on flight data. This module keeps the learned probe but
replaces the illegal field label with a reward built ENTIRELY from observed intensity.

The idea (see the "how does reward propagate" discussion). In plain M2 the probe p only
earns credit through a long chain -- p -> difference image d -> the policy reads d next
step -> a better correction -> lower predicted contrast H steps later. That weak signal
plateaus. Here we DECOUPLE the two halves of the action:

  * correction dc  -- keeps the control reward -||o'||^2 (dark-image energy), dense over
    the imagined rollout. Right signal for "null the field".
  * probe p        -- gets a one-step INFORMATION-GAIN reward: does seeing the difference
    image it produced let the policy pick a *better* next correction? Operationalised, in
    the model's imagination, as an ablation:

        r_info = ||o'(policy WITHOUT the difference image)||^2 - ||o'(policy WITH it)||^2

    If d carries real field information the policy corrects better with it than without, so
    r_info > 0; if d is noise the policy learns to ignore it and r_info -> 0. Every quantity
    is a model prediction of an *intensity* image -- no true field anywhere. The gradient
    reaches the probe through the model's differentiable d-prediction (the learned,
    self-supervised stand-in for EFC's hand-built Jacobian). Honest and in-flight-trainable.

Speed. Collection (20 steps x 3 CPU propagations) dominated wall time; this module (a) runs
data collection across N persistent worker processes (1 BLAS thread each -- more processes,
not more threads-per-process, is the right way to use cores for independent FFTs), and (b)
defaults max_steps to 8 (the absolute-action correction nulls on step 1; the tail is
refinement). Both are A/B-benchmarked by bench_probe_speed.py.
"""
from __future__ import annotations

import os
import argparse
import time
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.multiprocessing as mp

from po4ncpa import calibrate_per_mode_scale
from po4ncpa_probe import (
    ProbeConfig, ProbeDynamicsNet, ProbePolicyNet, ProbeBuffer, make_probe_env,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class InfoGainConfig(ProbeConfig):
    run_name: str = "po4ncpa_infogain"
    max_steps: int = 8                 # speed-up: absolute correction nulls fast; tail is refinement
    num_workers: int = 8               # parallel data-collection processes (1 BLAS thread each)
    lambda_info: float = 0.3           # weight of the probe info-gain reward vs the correction reward
    reward: str = "raw"                # "raw" (-||o'||^2) or "log" (-log10 energy; ~const grad/decade)
    lr_final_frac: float = 1.0         # linear LR decay of pol/dyn optimizers to this frac (1.0=off)
    explore_std_final: float = -1.0    # linear exploration-noise decay target (<0 = no decay)
    resume_refill: int = 20000         # on resume WITHOUT a saved buffer, collect this many TRANSITIONS before updates
    buffer_save_every: int = 10000     # periodically persist the replay buffer (episodes; 0=only at end, <0=never)
    resume_buffer: str = ""            # replay buffer to reload on resume ("" = auto: replay_buffer.npz next to --resume)


# ---------------------------------------------------------------------------
# Preprocessing + a single-episode rollout (shared by workers and eval)
# ---------------------------------------------------------------------------
def _pre(image_2ch: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    o = np.cbrt(image_2ch[0] - ideal)
    d = np.cbrt(image_2ch[1])
    return np.stack([o, d], axis=0).astype(np.float32)


def _run_episode(env, policy, ideal, A, explore_std, deterministic, seed, warmup, device):
    """Roll one episode; return (transitions, final_contrast, final_strehl).
    transitions is a list of (img[2,H,W], img_next[2,H,W], a_prev[A], a[A])."""
    obs, info = env.reset(seed=seed)
    img = _pre(obs["image"], ideal)
    a_prev = np.zeros(A, np.float32)
    trans = []
    last = info
    for _ in range(env.max_steps):
        if warmup:
            a = np.random.uniform(-1.0, 1.0, A).astype(np.float32)
        else:
            with torch.no_grad():
                it = torch.as_tensor(img[None], device=device)
                ap = torch.as_tensor(a_prev[None], device=device)
                a = policy(it, ap)[0].cpu().numpy().astype(np.float32)
            if not deterministic:
                a = np.clip(a + np.random.normal(0, explore_std, A), -1, 1).astype(np.float32)
        obs, _r, _t, _tr, info = env.step(a)
        img_next = _pre(obs["image"], ideal)
        if not deterministic:
            trans.append((img, img_next, a_prev, a))
        img, a_prev = img_next, a
        last = info
    return trans, last["contrast"], last["strehl"]


# ---------------------------------------------------------------------------
# Parallel collection: persistent spawn workers, each owns an env + a CPU policy copy
# ---------------------------------------------------------------------------
def _collect_worker(cfg, scale, probe_scale, task_q, res_q):
    torch.set_num_threads(1)                      # 1 thread/process -> no FFT oversubscription
    env = make_probe_env(cfg)
    env.action_scale = scale
    env.max_abs_actuator = scale
    env.probe_scale = probe_scale
    ideal = env.ideal_image().astype(np.float32)
    h, w = env.image_shape
    A = int(env.action_space.shape[0])
    policy = ProbePolicyNet(h, w, A, ch=cfg.ch)   # CPU
    policy.eval()
    while True:
        msg = task_q.get()
        if msg is None:
            break
        sd, seed, deterministic, warmup, explore_std = msg
        if sd is not None:
            policy.load_state_dict(sd)
        trans, c, s = _run_episode(env, policy, ideal, A, explore_std,
                                   deterministic, seed, warmup, "cpu")
        res_q.put((trans, c, s))


class ParallelCollector:
    def __init__(self, cfg, scale, probe_scale, num_workers):
        ctx = mp.get_context("spawn")
        self.task_q = ctx.Queue()
        self.res_q = ctx.Queue()
        self.ps = [ctx.Process(target=_collect_worker,
                               args=(cfg, scale, probe_scale, self.task_q, self.res_q),
                               daemon=True)
                   for _ in range(num_workers)]
        for p in self.ps:
            p.start()
        self.n = num_workers

    def dispatch(self, sd, seeds, deterministic, warmups, explore_std=0.1):
        """Non-blocking: queue len(seeds) episodes and return immediately. The workers run
        in their own processes, so the main process is free to do GPU updates meanwhile.
        Pair with gather(len(seeds)); the count is the caller's contract to drain them."""
        for i in range(len(seeds)):
            self.task_q.put((sd, seeds[i], deterministic, warmups[i], explore_std))

    def gather(self, n):
        """Block until n dispatched episodes have returned. Order not preserved."""
        return [self.res_q.get() for _ in range(n)]

    def collect(self, sd, seeds, deterministic, warmups, explore_std=0.1):
        """Dispatch len(seeds) episodes; block until all return. Order not preserved.
        (Convenience for eval/bench; the train loop uses dispatch/gather to overlap.)"""
        self.dispatch(sd, seeds, deterministic, warmups, explore_std)
        return self.gather(len(seeds))

    def close(self):
        for _ in self.ps:
            self.task_q.put(None)
        for p in self.ps:
            p.join(timeout=5)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class PO4NCPAInfoGain:
    def __init__(self, cfg: InfoGainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # A single main-process env for calibration, ideal image, shapes, and eval.
        self.env = make_probe_env(cfg)
        self.h, self.w = self.env.image_shape
        self.M = self.env.control_dim
        self.A = int(self.env.action_space.shape[0])         # 2M
        self.scale = calibrate_per_mode_scale(self.env.optics, cfg)
        self.probe_scale = cfg.probe_frac * self.scale
        self.env.action_scale = self.scale
        self.env.max_abs_actuator = self.scale
        self.env.probe_scale = self.probe_scale
        self.ideal = self.env.ideal_image().astype(np.float32)
        print(f"[infogain] correction scale min {self.scale.min():.2e} max {self.scale.max():.2e} | "
              f"probe scale (x{cfg.probe_frac}) min {self.probe_scale.min():.2e} "
              f"max {self.probe_scale.max():.2e} | modes={self.M} action={self.A}", flush=True)

        self.dyn = [ProbeDynamicsNet(self.h, self.w, self.A, ch=cfg.ch).to(self.device)
                    for _ in range(cfg.ensemble)]
        self.dyn_opt = [torch.optim.Adam(m.parameters(), lr=cfg.dyn_lr) for m in self.dyn]
        self.policy = ProbePolicyNet(self.h, self.w, self.A, ch=cfg.ch).to(self.device)
        self.pol_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.pol_lr)
        self.buf = ProbeBuffer(cfg.buffer, self.h, self.w, self.A)

        self.resumed = False
        self.buffer_loaded = False
        if cfg.resume:
            ck = torch.load(cfg.resume, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(ck["policy"])
            for m, sd in zip(self.dyn, ck["dynamics"]):
                m.load_state_dict(sd)
            if ck.get("pol_opt") is not None:
                self.pol_opt.load_state_dict(ck["pol_opt"])
            if ck.get("dyn_opt") is not None:
                for opt, sd in zip(self.dyn_opt, ck["dyn_opt"]):
                    opt.load_state_dict(sd)
            self.resumed = True
            print(f"[infogain] RESUMED from {cfg.resume}", flush=True)
            # Reload the replay buffer so the dynamics ensemble resumes on the SAME data
            # distribution it converged on -- eliminates the re-fit transient where the model
            # retrains on a freshly-refilled buffer while the policy chases the moving target.
            bpath = cfg.resume_buffer or os.path.join(os.path.dirname(cfg.resume),
                                                      "replay_buffer.npz")
            if os.path.exists(bpath):
                n = self.buf.load(bpath)
                self.buffer_loaded = True
                print(f"[infogain] reloaded replay buffer ({n} transitions) from {bpath} "
                      f"-> refill gate skipped, updates start immediately", flush=True)
            else:
                print(f"[infogain] no replay buffer at {bpath} -> refilling "
                      f"(resume_refill={cfg.resume_refill}); expect the usual resume transient",
                      flush=True)

        self.collector = ParallelCollector(cfg, self.scale, self.probe_scale, cfg.num_workers)

        self.log_dir = os.path.join("logs", cfg.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv = open(os.path.join(self.log_dir, "train.csv"), "w")
        self._csv.write("episode,train_contrast,train_strehl,eval_contrast,eval_strehl,info_gain\n")
        self._csv.flush()

    # --- policy state as CPU tensors for broadcast to workers ---------------
    def _sd_cpu(self):
        return {k: v.detach().cpu() for k, v in self.policy.state_dict().items()}

    def _dyn_rand(self, obs, ap, ac):
        return self.dyn[np.random.randint(self.cfg.ensemble)](obs, ap, ac)

    # --- dynamics: supervised on real transitions (predict both channels) --
    def update_dynamics(self):
        cfg = self.cfg
        losses = []
        for m, opt in zip(self.dyn, self.dyn_opt):
            for _ in range(cfg.dyn_iters):
                it, itn, ap, ac = self.buf.sample(cfg.batch, self.device)
                pred = m(it, ap, ac)
                num = torch.sqrt(((itn - pred) ** 2).sum(dim=[1, 2, 3]))
                den = torch.sqrt((itn ** 2).sum(dim=[1, 2, 3])) + 1e-8
                loss = (num / den).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0

    # --- policy: control reward on dc + info-gain reward on the probe -------
    def update_policy(self):
        cfg = self.cfg
        for p in (q for m in self.dyn for q in m.parameters()):
            p.requires_grad_(False)
        rewards, infos = [], []
        for _ in range(cfg.pol_iters):
            it, _itn, ap, _ac = self.buf.sample(cfg.batch, self.device)
            obs_t, a_tm1 = it, ap
            H = int(np.random.randint(cfg.h_min, cfg.h_max + 1))
            total_r, total_info = 0.0, 0.0
            for _k in range(H):
                m = self.dyn[np.random.randint(cfg.ensemble)]      # one member for the pair
                a = self.policy(obs_t, a_tm1)                       # [probe, correction]
                pred = m(obs_t, a_tm1, a)                          # [o', d']
                energy = (pred[:, 0:1] ** 2).sum(dim=[1, 2, 3])    # predicted dark-image energy
                # Control reward. "log" keeps the gradient ~constant per decade of contrast
                # (like EFC's residual-proportional pull), so the policy keeps digging near
                # the floor instead of stalling in the flat bottom of the raw -||o'||^2.
                r_corr = -torch.log10(energy + 1e-12) if cfg.reward == "log" else -energy

                # Info-gain = the MODEL's reliance on the difference channel of obs_t: how
                # much does its predicted post-correction dark image change when d is ablated
                # from the model input, holding the correction fixed (a detached)? Both
                # branches share the same action, so there is no policy-controlled
                # counterfactual to game (unlike a re-picked blind correction). obs_t's d was
                # produced by the PREVIOUS step's probe (k>=1), so the gradient flows only
                # into that probe -- a clean, non-hackable sensing signal. r_info>0 iff d
                # carries field information the (self-supervised) model actually uses.
                a_d = a.detach()
                obs_ab = obs_t.clone()
                obs_ab[:, 1:2] = 0.0
                o_with = m(obs_t, a_tm1, a_d)[:, 0:1]
                o_wo = m(obs_ab, a_tm1, a_d)[:, 0:1]
                r_info = ((o_with - o_wo) ** 2).sum(dim=[1, 2, 3])

                total_r = total_r + r_corr
                total_info = total_info + r_info
                obs_t = pred
                a_tm1 = a
            # Scale info to the control reward's magnitude so lambda_info is a true relative
            # weight (info := lambda x control-scale), robust to r_info's raw scale drifting
            # up as the model learns to use d. Refs are detached (weighting only, no grad).
            corr_ref = total_r.abs().mean().detach()
            info_ref = total_info.abs().mean().detach() + 1e-8
            total_info_scaled = total_info * (corr_ref / info_ref)
            loss = -(total_r + cfg.lambda_info * total_info_scaled).mean()
            self.pol_opt.zero_grad(); loss.backward(); self.pol_opt.step()
            rewards.append((total_r.mean() / H).item())
            infos.append((total_info.mean() / H).item())
        for p in (q for m in self.dyn for q in m.parameters()):
            p.requires_grad_(True)
        return (float(np.mean(rewards)) if rewards else 0.0,
                float(np.mean(infos)) if infos else 0.0)

    # --- eval: serial, deterministic, main-process GPU policy ---------------
    def evaluate(self):
        """Report the TRUE (noiseless) dark-hole contrast of the corrector state each
        episode lands on -- not the noisy measured contrast, which has a photon floor. Under
        flux=None this equals the measured contrast; under photon noise (M3) it isolates the
        controller's real performance from the measurement noise floor."""
        cs, ss = [], []
        for e in range(self.cfg.eval_episodes):
            _tr, _c, s = _run_episode(self.env, self.policy, self.ideal, self.A,
                                      self.cfg.explore_std, True, 100_000 + e, False, self.device)
            ct = float(self.env.optics.dark_hole_contrast())   # noiseless, at final correction
            cs.append(ct); ss.append(s)
        return float(np.median(cs)), float(np.median(ss))

    def _apply_schedules(self, ep):
        """Linear LR + exploration decay over total_episodes. Returns the current
        exploration std for this chunk's data collection."""
        cfg = self.cfg
        frac = min(ep / max(1, cfg.total_episodes), 1.0)
        if cfg.lr_final_frac != 1.0:
            mult = 1.0 + (cfg.lr_final_frac - 1.0) * frac        # 1 -> lr_final_frac
            for g in self.pol_opt.param_groups:
                g["lr"] = cfg.pol_lr * mult
            for opt in self.dyn_opt:
                for g in opt.param_groups:
                    g["lr"] = cfg.dyn_lr * mult
        if cfg.explore_std_final >= 0.0:
            return cfg.explore_std + (cfg.explore_std_final - cfg.explore_std) * frac
        return cfg.explore_std

    def train(self):
        cfg = self.cfg
        print(f"[infogain] device={self.device} image={self.h}x{self.w} modes={self.M} "
              f"action={self.A} workers={cfg.num_workers} max_steps={cfg.max_steps} "
              f"lambda_info={cfg.lambda_info} reward={cfg.reward} lr_final_frac={cfg.lr_final_frac} "
              f"explore_std_final={cfg.explore_std_final}", flush=True)
        t0 = time.time()
        best = -np.inf
        ep = 0
        last_eval = 0
        last_buf_save = 0

        def _plan(at_ep):
            """Chunk size + per-episode warmup flags + policy state for a dispatch at at_ep."""
            chunk = int(min(cfg.num_workers, cfg.total_episodes - at_ep))
            warmups = [((at_ep + i) < cfg.warmup_episodes) and not self.resumed
                       for i in range(chunk)]
            sd = None if all(warmups) else self._sd_cpu()
            return chunk, warmups, sd

        # Pipeline: dispatch collection -> run GPU updates while workers collect -> gather.
        # The workers live in their own processes, so the update phase (which was ~80% of
        # wall time and left the CPU idle) now overlaps the next round's CPU propagation.
        # The in-flight round collects with the policy from the previous round's updates --
        # one round of staleness, which is irrelevant for an off-policy replay buffer.
        explore_std_now = self._apply_schedules(ep)
        chunk, warmups, sd = _plan(ep)
        self.collector.dispatch(sd, [None] * chunk, False, warmups, explore_std_now)
        inflight = chunk

        while ep < cfg.total_episodes:
            # (a) GPU updates on the CURRENT buffer, concurrent with the in-flight collection.
            # On resume, keep the loaded (good) dynamics + policy frozen until the freshly
            # refilled buffer is representative -- avoids the transient where retraining the
            # model on a tiny buffer mismatches the resumed policy (chosen over saving the
            # ~3.6 GB replay buffer per checkpoint).
            updates_ok = self.buf.size >= cfg.batch
            if self.resumed and not self.buffer_loaded and self.buf.size < cfg.resume_refill:
                updates_ok = False
            dyn_loss = pol_r = info_r = 0.0
            if updates_ok:
                for _ in range(inflight):
                    dyn_loss = self.update_dynamics()
                    if (ep >= cfg.warmup_episodes) or self.resumed:
                        pol_r, info_r = self.update_policy()

            # (b) gather the round the workers were collecting during (a), add to buffer.
            c_tr = s_tr = 0.0
            for trans, c, s in self.collector.gather(inflight):
                for it, itn, ap, ac in trans:
                    self.buf.add(it, itn, ap, ac)
                c_tr, s_tr = c, s
            ep += inflight

            # (c) immediately dispatch the next round so it overlaps the next iter's updates.
            if ep < cfg.total_episodes:
                explore_std_now = self._apply_schedules(ep)
                chunk, warmups, sd = _plan(ep)
                self.collector.dispatch(sd, [None] * chunk, False, warmups, explore_std_now)
                inflight = chunk

            if (ep - last_eval) >= cfg.eval_every or ep >= cfg.total_episodes:
                ec, es = self.evaluate()
                last_eval = ep
                dt = time.time() - t0
                metric = -np.log10(max(ec, 1e-12)) if cfg.use_coronagraph else es
                print(f"ep {ep:5d} | dynL {dyn_loss:.3f} polR {pol_r:.3e} info {info_r:.3e} | "
                      f"train C {c_tr:.2e} S {s_tr:.3f} | EVAL C {ec:.3e} S {es:.4f} | "
                      f"buf {self.buf.size} | {dt:.0f}s", flush=True)
                self._csv.write(f"{ep},{c_tr:.6e},{s_tr:.6f},{ec:.6e},{es:.6f},{info_r:.6e}\n")
                self._csv.flush()
                if metric > best:
                    best = metric
                    self.save("best")

            # Periodically persist the replay buffer (separate ~3.6 GB file, overwritten) so a
            # future resume -- or a re-launch of a job killed here -- can reload the data
            # distribution and skip the refill transient. Decoupled from the frequent best
            # checkpoints to keep the write cost (a few seconds) rare.
            if cfg.buffer_save_every > 0 and (ep - last_buf_save) >= cfg.buffer_save_every \
                    and self.buf.size >= cfg.batch:
                last_buf_save = ep
                self.save_buffer()
        self.save("final")
        if cfg.buffer_save_every >= 0 and self.buf.size >= cfg.batch:
            self.save_buffer()
        self._csv.close()
        self.collector.close()
        print(f"[infogain] done in {time.time()-t0:.0f}s", flush=True)

    def save(self, tag):
        path = os.path.join(self.log_dir, f"po4ncpa_infogain_{tag}.pt")
        torch.save({"policy": self.policy.state_dict(),
                    "dynamics": [m.state_dict() for m in self.dyn],
                    "pol_opt": self.pol_opt.state_dict(),
                    "dyn_opt": [o.state_dict() for o in self.dyn_opt],
                    "cfg": vars(self.cfg)}, path)

    def save_buffer(self):
        """Write the replay buffer to logs/<run>/replay_buffer.npz atomically (tmp + rename),
        so a resume that dies mid-write never reads a truncated file."""
        final = os.path.join(self.log_dir, "replay_buffer.npz")
        tmp = os.path.join(self.log_dir, "replay_buffer.tmp.npz")
        t = time.time()
        self.buf.save(tmp)
        os.replace(tmp, final)
        print(f"[infogain] saved replay buffer ({self.buf.size} transitions) -> {final} "
              f"({time.time()-t:.1f}s)", flush=True)


def parse_args() -> InfoGainConfig:
    p = argparse.ArgumentParser(description="PO4NCPA + observable-only info-gain probe (M2-honest)")
    d = InfoGainConfig()
    for f in fields(d):
        name = "--" + f.name.replace("_", "-")
        if isinstance(getattr(d, f.name), bool):
            p.add_argument(name, type=lambda x: x.lower() in ("1", "true", "yes"),
                           default=getattr(d, f.name))
        else:
            p.add_argument(name, type=type(getattr(d, f.name)), default=getattr(d, f.name))
    a = p.parse_args()
    return InfoGainConfig(**{f.name: getattr(a, f.name) for f in fields(d)})


if __name__ == "__main__":
    PO4NCPAInfoGain(parse_args()).train()
