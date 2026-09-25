"""Before/after speed benchmark for the info-gain trainer.

Isolates the two collection-side speed-ups -- parallel workers and shorter episodes
(max_steps) -- and reports collection throughput plus the per-episode GPU update cost so
the total wall-time-per-episode of the baseline vs the optimized config can be compared.

Four collection configs are timed:
    A  1 worker,  max_steps=20   (baseline, ~serial + minor IPC)
    B  1 worker,  max_steps=8
    C  W workers, max_steps=20
    D  W workers, max_steps=8    (optimized)

Runs on the GPU partition via slurm/bench_probe_speed.slurm (builds CoronagraphOptics, so
never on the login node)."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from po4ncpa_infogain import InfoGainConfig, PO4NCPAInfoGain


def time_collection(cfg, n_episodes, warm=1):
    """Build a trainer with cfg, warm the workers, then time n_episodes of collection.
    Returns seconds/episode (collection only)."""
    tr = PO4NCPAInfoGain(cfg)
    sd = tr._sd_cpu()
    # warm: pay worker JIT / first-FFT costs once; keep the transitions so the buffer has
    # data for the update-timing block below.
    nwarm = min(warm * cfg.num_workers, n_episodes)
    warm_res = tr.collector.collect(sd, [None] * nwarm, False, [False] * nwarm, cfg.explore_std)
    for trans, _c, _s in warm_res:
        for item in trans:
            tr.buf.add(*item)
    t0 = time.time()
    done = 0
    while done < n_episodes:
        chunk = min(cfg.num_workers, n_episodes - done)
        tr.collector.collect(sd, [None] * chunk, False, [False] * chunk, cfg.explore_std)
        done += chunk
    dt = time.time() - t0
    # per-episode GPU update cost: fill buffer from the warm episodes already collected,
    # then time a handful of full update rounds (dynamics + policy).
    upd = float("nan")
    if tr.buf.size >= cfg.batch:
        n_upd = 10
        t1 = time.time()
        for _ in range(n_upd):
            tr.update_dynamics()
            tr.update_policy()
        upd = (time.time() - t1) / n_upd
    tr.collector.close()
    return dt / n_episodes, upd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--episodes", type=int, default=28)
    args = ap.parse_args()

    base = dict(total_episodes=10 ** 9, warmup_episodes=0, eval_every=10 ** 9,
                ensemble=5, ch=64, run_name="bench_tmp")

    configs = [
        ("A serial  max20", 1, 20),
        ("B serial  max8 ", 1, 8),
        ("C par     max20", args.workers, 20),
        ("D par     max8 ", args.workers, 8),
    ]

    print(f"# benchmark: {args.episodes} episodes/config, {args.workers} workers for parallel\n")
    rows = []
    for name, nw, ms in configs:
        cfg = InfoGainConfig(num_workers=nw, max_steps=ms, **base)
        # buffer needs some transitions to time updates; keep it small/fast
        sec_ep, sec_upd = time_collection(cfg, args.episodes)
        rows.append((name, nw, ms, sec_ep, sec_upd))
        print(f"{name} | workers {nw:2d} | max_steps {ms:2d} | "
              f"collect {sec_ep:6.3f} s/ep | update {sec_upd:6.3f} s/ep", flush=True)

    a_collect = rows[0][3]
    a_upd = rows[0][4]
    d_collect = rows[-1][3]
    d_upd = rows[-1][4]
    a_total = a_collect + (a_upd if a_upd == a_upd else 0.0)   # NaN-safe
    d_total = d_collect + (d_upd if d_upd == d_upd else 0.0)
    print("\n# summary")
    print(f"collection speedup A->D: {a_collect / d_collect:6.2f}x "
          f"({a_collect:.3f} -> {d_collect:.3f} s/ep)")
    if a_total > 0 and d_total > 0:
        print(f"total (collect+update) A->D: {a_total / d_total:6.2f}x "
              f"({a_total:.3f} -> {d_total:.3f} s/ep)")
        print(f"episodes/hour A: {3600 / a_total:8.0f}   D: {3600 / d_total:8.0f}")


if __name__ == "__main__":
    main()
