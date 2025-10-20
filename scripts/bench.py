#!/usr/bin/env python
import argparse, os, subprocess, sys, shlex, pathlib

def main():
    ap = argparse.ArgumentParser(description="Run a single pipeline bench config.")
    ap.add_argument("--schedule", choices=["gpipe","1f1b","interleaved"], default="gpipe")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--microbatches", type=int, default=2)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--measure", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={args.procs}",
        "scripts/pp_runner.py",
        "--schedule", args.schedule,
        "--n-layers", str(args.layers),
        "--n-heads", str(args.heads),
        "--dim", str(args.dim),
        "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch),
        "--microbatches", str(args.microbatches),
        "--threads", str(args.threads),
        "--warmup-steps", str(args.warmup),
        "--measure-steps", str(args.measure),
        "--emit-csv",
    ]
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    print(">>", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd, env=env)

if __name__ == "__main__":
    pathlib.Path("results").mkdir(parents=True, exist_ok=True)
    sys.exit(main())
