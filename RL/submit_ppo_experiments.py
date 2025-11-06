#!/usr/bin/env python
"""
submit_ppo_experiments.py

Generate and (optionally) submit multiple SLURM jobs by sweeping parameter grids
for train_ppo.py. Uses RL/rl_job.slurm as a template and writes job files into
RL/jobs/<timestamp>/.

Features:
- Parameter grid expansion (total_timesteps, num_envs, num_modes, pixels, oversizing, num_airy, ppsr, eval_freq, seed, include_slopes)
- Per-run log directories (logs/<timestamp>/<tag>/) and tb_log_name derived from tag
- Auto job-name and output/error file naming based on params
- Dry-run mode to just generate files without submission
- Optional immediate submission with sbatch (if running on cluster)

Example:
  python submit_ppo_experiments.py \
    --total_timesteps 200000 1000000 \
    --num_modes 4 5 \
    --include_slopes true false \
    --eval_freq 10000 \
    --time 08:00:00 \
    --submit
"""
from __future__ import annotations
import argparse
import itertools
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "rl_job.slurm"
JOBS_ROOT = Path(__file__).parent / "jobs"
LOGS_ROOT = Path(__file__).parent / "logs"


def read_template() -> str:
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_job_text(template: str,
                   job_name: str,
                   time_limit: str,
                   stdout_path: str,
                   stderr_path: str,
                   conda_env_activate_line: str,
                   exec_line: str) -> str:
    lines = []
    for ln in template.splitlines():
        if ln.startswith("#SBATCH --job-name="):
            ln = f"#SBATCH --job-name={job_name}"
        elif ln.startswith("#SBATCH --time="):
            ln = f"#SBATCH --time={time_limit}"
        elif ln.startswith("#SBATCH --output="):
            ln = f"#SBATCH --output={stdout_path}"
        elif ln.startswith("#SBATCH --error="):
            ln = f"#SBATCH --error={stderr_path}"
        elif ln.strip().startswith("conda activate"):
            ln = conda_env_activate_line
        elif ln.strip().startswith("python train_ppo.py"):
            ln = exec_line
        lines.append(ln)
    # Ensure we have output/error lines
    if not any(l.startswith("#SBATCH --output=") for l in lines):
        lines.insert(2, f"#SBATCH --output={stdout_path}")
    if not any(l.startswith("#SBATCH --error=") for l in lines):
        lines.insert(3, f"#SBATCH --error={stderr_path}")
    # Replace module/activate if not present
    if not any(l.strip().startswith("conda activate") for l in lines):
        # Try to insert after module load
        for idx, l in enumerate(lines):
            if l.strip().startswith("module load"):
                lines.insert(idx + 1, conda_env_activate_line)
                break
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Submit grid of train_ppo.py experiments via SLURM")
    # PPO/train params to sweep
    ap.add_argument("--total_timesteps", nargs="+", type=int, default=[200_000])
    ap.add_argument("--num_envs", nargs="+", type=int, default=[1])
    ap.add_argument("--num_modes", nargs="+", type=int, default=[4])
    ap.add_argument("--basis", nargs="+", type=str, default=["zernike"])
    ap.add_argument("--pixels", nargs="+", type=int, default=[64])
    ap.add_argument("--oversizing_factor", nargs="+", type=float, default=[1.0])
    ap.add_argument("--num_airy", nargs="+", type=int, default=[5])
    ap.add_argument("--ppsr", nargs="+", type=int, default=[2])
    ap.add_argument("--eval_freq", nargs="+", type=int, default=[5000])
    ap.add_argument("--seed", nargs="+", type=int, default=[42])
    ap.add_argument("--include_slopes", nargs="+", choices=["true", "false"], default=["true"],
                    help="Whether to include slopes in observations: true|false")

    # SLURM overrides
    ap.add_argument("--time", default="08:00:00")
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--mem", default="8G")
    ap.add_argument("--gres", default="gpu:1")
    ap.add_argument("--mail_user", default=None)
    ap.add_argument("--job_prefix", default="ppo")

    # Python env activation line (cluster specific)
    ap.add_argument("--conda_activate", default="conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2")

    # Submission control
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = JOBS_ROOT / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    logs_timestamp_root = LOGS_ROOT / timestamp
    logs_timestamp_root.mkdir(parents=True, exist_ok=True)

    template = read_template()

    combos = list(itertools.product(
        args.total_timesteps,
        args.num_envs,
        args.num_modes,
        args.basis,
        args.pixels,
        args.oversizing_factor,
        args.num_airy,
        args.ppsr,
        args.eval_freq,
        args.seed,
        args.include_slopes,
    ))

    generated_files = []

    for total_timesteps, num_envs, num_modes, basis, pixels, oversizing_factor, num_airy, ppsr, eval_freq, seed, include_slopes_str in combos:
        include_slopes_bool = (include_slopes_str.lower() == "true")
        slopes_tag = "slopes" if include_slopes_bool else "noslopes"

        # Build a concise tag
        tag = (
            f"{args.job_prefix}-tm{total_timesteps}-ne{num_envs}-m{num_modes}-px{pixels}-ov{oversizing_factor}-"
            f"na{num_airy}-q{ppsr}-ef{eval_freq}-s{seed}-{slopes_tag}"
        )
        job_name = tag
        stdout_path = str(out_root / f"{tag}.out")
        stderr_path = str(out_root / f"{tag}.err")

        # Per-run log dir and tb name
        log_dir = logs_timestamp_root / tag
        os.makedirs(log_dir, exist_ok=True)
        tb_log_name = tag

        # Build python exec line for train_ppo.py
        exec_parts = [
            "python train_ppo.py",
            f"--total-timesteps {total_timesteps}",
            f"--num-envs {num_envs}",
            f"--num-modes {num_modes}",
            f"--basis {basis}",
            f"--pixels {pixels}",
            f"--oversizing-factor {oversizing_factor}",
            f"--num-airy {num_airy}",
            f"--ppsr {ppsr}",
            f"--log-dir {shlex.quote(str(log_dir))}",
            f"--tb-log-name {shlex.quote(tb_log_name)}",
            f"--eval-freq {eval_freq}",
            f"--seed {seed}",
        ]
        # Explicit include/no-slopes flag for clarity
        exec_parts.append("--include-slopes" if include_slopes_bool else "--no-slopes")
        exec_line = " ".join(exec_parts)

        job_text = build_job_text(
            template=template,
            job_name=job_name,
            time_limit=args.time,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            conda_env_activate_line=args.conda_activate,
            exec_line=exec_line,
        )

        # Patch other sbatch options if present
        lines = job_text.splitlines()
        def replace_or_append(prefix: str, new_line: str):
            for i, ln in enumerate(lines):
                if ln.startswith(prefix):
                    lines[i] = new_line
                    return
            lines.insert(2, new_line)
        replace_or_append("#SBATCH --cpus-per-task=", f"#SBATCH --cpus-per-task={args.cpus}")
        replace_or_append("#SBATCH --gres=", f"#SBATCH --gres={args.gres}")
        if args.mail_user:
            replace_or_append("#SBATCH --mail-user=", f"#SBATCH --mail-user={args.mail_user}")
        job_text = "\n".join(lines) + "\n"

        job_path = out_root / f"{tag}.slurm"
        with open(job_path, "w", encoding="utf-8") as f:
            f.write(job_text)
        generated_files.append(job_path)

        if args.submit and not args.dry_run:
            try:
                print(f"Submitting {job_path} ...")
                subprocess.run(["sbatch", str(job_path)], check=True)
            except Exception as e:
                print(f"Failed to submit {job_path}: {e}")

    print(f"Generated {len(generated_files)} jobs under {out_root}")
    print(f"Logs will be written under {logs_timestamp_root}")
    if not args.submit:
        print("Dry run or generation only. Use --submit to submit to SLURM.")


if __name__ == "__main__":
    main()
