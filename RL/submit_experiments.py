#!/usr/bin/env python
"""
submit_experiments.py

Generate and (optionally) submit multiple SLURM jobs by sweeping parameter grids
for train_job.py. Uses RL/train_job.slurm as a template and writes job files into
RL/jobs/<timestamp>/.

Features:
- Parameter grid expansion (model_type, norm, epochs, batch_size, lr, out_dir suffixes, etc.)
- Auto job-name and output/error file naming based on params
- Dry-run mode to just generate files without submission
- Optional immediate submission with sbatch (if running on cluster)

Example:
  python submit_experiments.py \
    --datapath /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/data/dataset100K25M.pkl \
    --model_type fc1 cnn1 \
    --norm minmax zscore log \
    --epochs 200 500 \
    --batch_size 256 512 \
    --lr 1e-3 3e-4 \
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

TEMPLATE_PATH = Path(__file__).parent / "train_job.slurm"
JOBS_ROOT = Path(__file__).parent / "jobs"


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
        elif ln.strip().startswith("python train_job.py"):
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
    ap = argparse.ArgumentParser(description="Submit grid of train_job.py experiments via SLURM")
    ap.add_argument("--datapath", required=True)
    ap.add_argument("--model_type", nargs="+", default=["fc1"])
    ap.add_argument("--norm", nargs="+", default=["minmax"])  # minmax|zscore|log
    ap.add_argument("--epochs", nargs="+", type=int, default=[10000])
    ap.add_argument("--batch_size", nargs="+", type=int, default=[2048])
    ap.add_argument("--lr", nargs="+", default=["1e-3"])  # keep as str for readability in names
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", nargs="+", type=int, default=[42])
    # Train input type grid: images or slopes
    ap.add_argument(
        "--train_type",
        nargs="+",
        default=["images"],
        help="One or more train input types to sweep: images or slopes"
    )
    # Data cutoff sweep (first N samples). Use -1 to indicate full dataset.
    ap.add_argument(
        "--data_cutoff",
        nargs="+",
        type=int,
        default=[-1],
        help="One or more cutoff sizes (first N samples). Use -1 for all samples."
    )
    # TF1 (ViT) sweep parameters (only applied when model_type includes 'tf1')
    ap.add_argument("--tf1_patch_size", nargs="+", type=int, default=[8])
    ap.add_argument("--tf1_dim", nargs="+", type=int, default=[128])
    ap.add_argument("--tf1_depth", nargs="+", type=int, default=[4])
    ap.add_argument("--tf1_heads", nargs="+", type=int, default=[4])
    ap.add_argument("--tf1_mlp_dim", nargs="+", type=int, default=[256])
    ap.add_argument("--tf1_attn_dropout", nargs="+", type=float, default=[0.0])
    ap.add_argument("--tf1_emb_dropout", nargs="+", type=float, default=[0.0])
    # CLS token control: 'on' to include flag, 'off' to omit flag (mean pool)
    ap.add_argument("--tf1_use_cls_token", nargs="+", choices=["on", "off"], default=["on"],
                    help="Sweep whether to use CLS token (on) or mean-pool (off)")
    # FC1 architecture override (can accept multiple specs to grid over)
    # Example: --fc1_hidden 256,128 512,256,128
    ap.add_argument(
        "--fc1_hidden",
        nargs="*",
        default=[],
        help="One or more comma-separated hidden layer specs for FC1 (e.g., '256,128' '512,256,128')."
    )

    ap.add_argument("--split_vector", action="store_true", help="Use vector split loss (direction + magnitude)")
    ap.add_argument("--cpu_only", action="store_true", help="Force CPU-only training (omit GPU resources and add --cpu_only to train_job.py).")

    # SLURM overrides
    ap.add_argument("--time", default="08:00:00")
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--mem", default="8G")
    ap.add_argument("--gres", default="gpu:1")
    ap.add_argument("--mail_user", default=None)
    ap.add_argument("--job_prefix", default="exp")

    # Python env activation line (cluster specific)
    ap.add_argument("--conda_activate", default="conda activate /scratch/network/ak9088/anaconda3/envs/dmrl2")

    # Submission control
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = JOBS_ROOT / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    template = read_template()

    # Build full cartesian product, including TF1 params (ignored for non-tf1 models)
    combos = list(itertools.product(
        args.model_type,
        args.norm,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
        args.train_type,
        args.data_cutoff,
        args.tf1_patch_size,
        args.tf1_dim,
        args.tf1_depth,
        args.tf1_heads,
        args.tf1_mlp_dim,
        args.tf1_attn_dropout,
        args.tf1_emb_dropout,
        args.tf1_use_cls_token,
    ))

    generated_files = []

    for (
        model_type, norm, epochs, batch_size, lr, seed, train_type, data_cutoff,
        tf1_ps, tf1_dim, tf1_depth, tf1_heads, tf1_mlp, tf1_attn_do, tf1_emb_do, tf1_cls_flag
    ) in combos:
        # Determine FC1 hidden-layer sweep values (only applies to fc1); for others, single None
        fc1_hidden_specs = args.fc1_hidden if (model_type == "fc1" and len(args.fc1_hidden) > 0) else [None]

        for fc1_hidden in fc1_hidden_specs:
            # Tag augmentation for FC1 hidden spec
            if model_type == "fc1" and fc1_hidden:
                clean = "-h" + fc1_hidden.replace(",", "-").replace(" ", "")
            else:
                clean = ""
            tt_suffix = f"-{train_type}"
            tf_suffix = ""
            if model_type == "tf1":
                # concise descriptor for transformer settings
                cls_suffix = "-cls" if tf1_cls_flag == "on" else "-mean"
                tf_suffix = f"-ps{tf1_ps}-d{tf1_dim}-L{tf1_depth}-h{tf1_heads}-mlp{tf1_mlp}{cls_suffix}"
            dc_suffix = f"-dc{data_cutoff}" if (data_cutoff is not None and data_cutoff > 0) else ""
            tag = f"{args.job_prefix}-{model_type}-{norm}-ep{epochs}-bs{batch_size}-lr{lr}-s{seed}{clean}{tt_suffix}{tf_suffix}{dc_suffix}"
            job_name = tag
            stdout_path = str(out_root / f"{tag}.out")
            stderr_path = str(out_root / f"{tag}.err")
            out_dir = str(out_root / tag)
            os.makedirs(out_dir, exist_ok=True)

            # Build python exec line
            exec_parts = [
                "python train_job.py",
                f"--datapath {shlex.quote(args.datapath)}",
                f"--model_type {model_type}",
                f"--norm {norm}",
                f"--epochs {epochs}",
                f"--batch_size {batch_size}",
                f"--lr {lr}",
                f"--val_split {args.val_split}",
                f"--seed {seed}",
                f"--train_type {train_type}",
                f"--out_dir {shlex.quote(out_dir)}",
            ]
            if args.split_vector:
                exec_parts.append(f"--split_vector")
            if data_cutoff is not None and data_cutoff > 0:
                exec_parts.append(f"--data_cutoff {data_cutoff}")
            # Optional TF1 passthrough when applicable
            if model_type == "tf1":
                exec_parts.extend([
                    f"--tf1_patch_size {tf1_ps}",
                    f"--tf1_dim {tf1_dim}",
                    f"--tf1_depth {tf1_depth}",
                    f"--tf1_heads {tf1_heads}",
                    f"--tf1_mlp_dim {tf1_mlp}",
                    f"--tf1_attn_dropout {tf1_attn_do}",
                    f"--tf1_emb_dropout {tf1_emb_do}",
                ])
                if tf1_cls_flag == "on":
                    exec_parts.append("--tf1_use_cls_token")
            # Optional FC1 architecture passthrough
            if model_type == "fc1" and fc1_hidden:
                hidden_spec = fc1_hidden.replace(",", " ").strip()
                if hidden_spec:
                    exec_parts.append(f"--fc1_hidden {hidden_spec}")
            if args.cpu_only:
                exec_parts.append("--cpu_only")
            exec_line = " ".join(exec_parts)

            # Allow SLURM header overrides (cpus, mem, gres, time, mail)
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
            replace_or_append("#SBATCH --mem=", f"#SBATCH --mem={args.mem}")
            if not args.cpu_only:
                replace_or_append("#SBATCH --gres=", f"#SBATCH --gres={args.gres}")
            else:
                # Remove any existing GPU gres line if template had one
                lines = [l for l in lines if not l.startswith("#SBATCH --gres=")]
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
    if not args.submit:
        print("Dry run or generation only. Use --submit to submit to SLURM.")


if __name__ == "__main__":
    main()
