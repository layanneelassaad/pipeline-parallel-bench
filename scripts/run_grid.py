import subprocess, sys, itertools, csv, pathlib, time, os, shlex
import pathlib
THIS_DIR = pathlib.Path(__file__).parent
PP = str((THIS_DIR / "pp_runner.py").resolve())
PP = "pp_runner.py"
LAYERS_LIST = [4,8,12]
HEADS_LIST = [4,8,12]
PROCS_LIST = [2,4]
SCHEDULES = ["gpipe","1f1b","interleaved"]

SEQ_LEN = 64
BATCH = 8
DIM = 240
MICRO_PER_PROCS = {2:4, 4:8}

THREADS = 1
WARMUP = 0
MEASURE = 1
TIMEOUT_S = 120

OUT_PATH = pathlib.Path("results/pp_results_fast.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def ensure_header(path):
    if not path.exists():
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "schedule","world_size","layers","heads","hidden_dim",
                "seq_len","batch_size","microbatches","threads",
                "avg_step_time_s","total_time_s","tokens_per_s",
                "timestamp"
            ])

def parse_csv_line(line):
    r = next(csv.reader([line]))
    return {
        "schedule": r[0],
        "world_size": int(r[1]),
        "layers": int(r[2]),
        "heads": int(r[3]),
        "hidden_dim": int(r[4]),
        "seq_len": int(r[5]),
        "batch_size": int(r[6]),
        "microbatches": int(r[7]),
        "threads": int(r[8]),
        "avg_step_time_s": float(r[9]),
        "total_time_s": float(r[10]),
        "tokens_per_s": float(r[11]),
    }

def append_row(path, row):
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            row["schedule"],row["world_size"],row["layers"],row["heads"],row["hidden_dim"],
            row["seq_len"],row["batch_size"],row["microbatches"],row["threads"],
            row["avg_step_time_s"],row["total_time_s"],row["tokens_per_s"],
            int(time.time())
        ])

def run_once(nproc, schedule, layers, heads):
    micro = MICRO_PER_PROCS[nproc]
    cmd = [
        "torchrun","--standalone",f"--nproc_per_node={nproc}", PP,
        "--schedule", schedule,
        "--n-layers", str(layers),
        "--n-heads", str(heads),
        "--dim", str(DIM),
        "--seq-len", str(SEQ_LEN),
        "--batch-size", str(BATCH),
        "--microbatches", str(micro),
        "--threads", str(THREADS),
        "--warmup-steps", str(WARMUP),
        "--measure-steps", str(MEASURE),
        "--emit-csv",
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout after {TIMEOUT_S}s: {' '.join(shlex.quote(c) for c in cmd)}")
    out = (p.stdout or "") + (p.stderr or "")
    line = None
    for ln in out.splitlines():
        if ln.count(",") >= 10 and (ln.startswith("gpipe") or ln.startswith("1f1b") or ln.startswith("interleaved")):
            line = ln.strip()
    if line is None:
        raise RuntimeError(f"No CSV line from torchrun.\nCommand: {' '.join(cmd)}\n--- OUT ---\n{out}\n-----------")
    return parse_csv_line(line)

def main():
    assert all(DIM % h == 0 for h in HEADS_LIST), f"DIM={DIM} must be divisible by all HEADS_LIST={HEADS_LIST}"
    ensure_header(OUT_PATH)
    total = len(LAYERS_LIST)*len(HEADS_LIST)*len(PROCS_LIST)*len(SCHEDULES)
    done = 0
    start = time.time()
    for layers, heads, procs in itertools.product(LAYERS_LIST, HEADS_LIST, PROCS_LIST):
        for schedule in SCHEDULES:
            row = run_once(procs, schedule, layers, heads)
            append_row(OUT_PATH, row)
            done += 1
            elapsed = time.time() - start
            print(f"[{done}/{total}] OK schedule={schedule} procs={procs} layers={layers} heads={heads} tps={row['tokens_per_s']:.2f} elapsed_s={elapsed:.1f}", flush=True)

if __name__ == "__main__":
    main()
