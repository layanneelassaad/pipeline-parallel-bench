# Pipeline Parallelism Benchmarks (PyTorch, CPU)

This repo compares **GPipe**, **1F1B**, and **Interleaved-1F1B** pipeline schedules using PyTorch’s
`torch.distributed.pipelining` with **CPU distributed training**. It provides a CLI runner, a small grid sweep,
and an analysis script that turns CSV into plots for **speedup vs GPipe** and **scaling efficiency**.


---

## Pipeline parallelism and pipeline bubbles:

**Pipeline parallelism** does two things:

1) **Stage the model** – split the network into **K** sequential stages and assign consecutive layers to ranks (GPUs/CPUs) `0..K-1`.  
   Each stage runs forward/backward for its layers only.

2) **Split the global batch** into **M** micro-batches and **inject them sequentially** through the pipeline.

**Pipeline bubbles** are the idle gaps when a stage can’t do useful work (e.g., waiting for inputs/gradients). Lower utilization ⇒ lower throughput.

Let one “tick” be the time for one stage to do **one forward or backward** on **one micro-batch**.

- **Forward fill bubble** (≈ K−1 ticks): At `t=0` only stage 0 works on μ₁; others wait. The tail stage starts only after μ₁ traverses the earlier stages.
- **Forward drain bubble** (≈ K−1 ticks): After stage 0 finishes μ_M, it goes idle; the remaining stages finish μ_M one by one.
- **Backward fill bubble** (≈ K−1 ticks): Backward starts at the tail on μ_M; upstream stages are idle until gradients arrive.
- **Backward drain bubble** (≈ K−1 ticks): As gradients for μ₁ climb back, tail stages go idle one by one until stage 0 finishes.

These bubbles add a roughly **O(K)** fixed cost per mini-batch. Schedules differ in how they **overlap** work and **bound memory** to reduce bubble impact.

---

## Schedules & trade-offs

### GPipe
- **Idea:** Use many micro-batches (**M ≫ K**) so the pipeline fills quickly and stays full.  
- **Pros:** Simple and good utilization once warmed up.  
- **Cons (memory):** Each stage must retain **activations for all M micro-batches** until their backward arrives → **activation memory O(M)**.

### 1F1B (One-Forward-One-Backward)
- **Idea:** After warm-up, **alternate** forward/backward at each stage. As soon as the tail finishes forward on μ₁, it immediately starts backward on μ₁.  
- **Effect on bubbles:** Overlaps the middle gaps—no large “forward drain” or “backward fill”; only the initial forward fill and final backward drain remain.  
- **Memory:** Each activation is held for at most ~the pipeline depth (forward trip + backward return) → **activation memory O(K)**.

### Interleaved-1F1B (with virtual stages)
- **Idea:** Split each stage into **v virtual chunks**, creating **K·v shorter stages**. Micro-batches traverse chunk0 of all ranks, then chunk1 of all ranks, etc.  
- **Effect on bubbles:** **Fill/drain time scales down by ~1/v** (shorter stage latency), so total bubble time shrinks proportionally.  
- **Memory:** Retains the **O(K)** bound from 1F1B (bounded by pipeline depth, not M).  
- **Trade-off:** More cross-rank hand-offs (≈ v× more) → higher **communication / context-switch overhead**.

---

## Why compare on **CPU distributed training**?

- We use **`gloo`** backend with **multiple CPU processes** (via `torchrun`), so results capture schedule overheads **without** GPU/NCCL effects.
- On CPU, **communication and synchronization** are relatively expensive; **compute per stage** is smaller.  
  This shifts sweet spots (e.g., interleaving may help at **P=2** for wider models but not at **P=4**, where overhead dominates).  
- On GPUs with **NCCL** and larger compute per stage, crossover points can move (1F1B/interleaving often win more broadly).

---

## Setup:
> **Note:** `scripts/analyze_grid.py` expects a results CSV at `results/pp_results_fast.csv`.  
> If you don’t have one yet, either run the grid (see below) or copy a CSV into `results/`.

Option 1: local run
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# PyTorch CPU:
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.*" torchvision torchaudio || \
pip install "torch==2.8.*" torchvision torchaudio
pip install -r requirements.txt
python scripts/analyze_grid.py
```

Sanity command:
```bash
python -m torchrun --standalone --nproc_per_node=2 scripts/pp_runner.py \
  --schedule gpipe --n-layers 4 --n-heads 4 --dim 240 \
  --seq-len 64 --batch-size 8 --microbatches 4 \
  --threads 1 --warmup-steps 0 --measure-steps 1 --emit-csv
```


Option 2: Docker (build locally):
Make sure Docker Desktop is running (macOS/Windows) or run `colima start` (Homebrew).
```bash
docker build -t pp-bench:cpu .
docker run --rm -it pp-bench:cpu
docker run --rm -it -v "$PWD/results:/app/results" pp-bench:cpu \
  python scripts/analyze_grid.py
```

Option 3: pull a prebuilt Docker image 
```bash
docker pull layanneelassaad/pp-bench:cpu-v0.1 
docker run --rm -it layanneelassaad/pp-bench:cpu-v0.1
docker run --rm -it -v "$PWD/results:/app/results" layanneelassaad/pp-bench:cpu-v0.1 \
  python scripts/analyze_grid.py
```
---

## Results:
| Model | Procs (P) |      GPipe |       1F1B | Interleaved | Winner          |
| ----- | --------: | ---------: | ---------: | ----------: | --------------- |
| 4L-4H |         2 |     623.89 | **749.93** |      710.74 | **1F1B**        |
| 4L-4H |         4 |     878.28 | **916.92** |      873.83 | **1F1B**        |
| 4L-8H |         2 |     729.22 |     703.81 |  **749.00** | **Interleaved** |
| 4L-8H |         4 | **924.42** |     915.20 |      891.74 | **GPipe**       |


Scaling efficiency (selected):

P=2 ≈ 50–60%

P=4 ≈ 24–27%

## Analysis (CPU / gloo)

<p align="center">
  <img alt="Speedup vs GPipe" src="results/speedup_vs_gpipe.png" width="49%">
 
</p>

**Figure: Speedup vs GPipe.**  
For the **4L-4H** model at **P=2**, **1F1B** delivers about **1.20×** GPipe while **Interleaved** is ~**1.13×**. This matches the intuition that when each pipeline stage is relatively cheap, alternating forward/backward quickly erases the middle bubbles and keeps ranks busy with minimal coordination cost. At **P=4** on CPU, the advantage narrows: **1F1B** is only ~**1.04×**, and **Interleaved** drops to parity (≈**1.00×**). The extra barriers and hand-offs introduced at higher process counts eat into the gains that overlap provides.
For the **4L-8H** model, widening increases per-stage compute a bit. At **P=2**, **Interleaved** edges ahead at ~**1.03×** GPipe, while **1F1B** dips slightly below (**~0.96×**). Splitting each stage into virtual chunks keeps the pipeline better utilized at small P by shortening stage latency and reducing the visible bubble. However, at **P=4** both **1F1B** (~**0.99×**) and **Interleaved** (~**0.97×**) fall behind GPipe on CPU; the extra cross-rank transfers (v× more for Interleaved) plus more synchronization dominate at larger P.

On CPU, **1F1B** is best for **small/deep-ish** configs at **P=2** and still slightly ahead at **P=4** when stages are cheap (4L-4H). **Interleaved** can win at **P=2** for **wider** models (4L-8H) by adding virtual stages, but at **P=4** its extra hand-offs negate the benefit; **GPipe** becomes competitive or best.

---
<p align="center">
 <img alt="Scaling Efficiency" src="results/scaling_efficiency.png" width="49%">
 </p>
 
**Figure: Scaling efficiency (%).**  
- **P=2:** Winners sit ~**50–60%** efficient (e.g., 4L-4H with 1F1B ≈ 60%). Wider 4L-8H models are slightly lower (~**48–51%**), reflecting more per-stage compute.  
- **P=4:** Efficiency drops to ~**24–27%** across schedules. With `gloo` on CPU, tokens/s still rises, but barrier latency, context switches, and tensor hand-offs grow faster than useful work per stage.

Going from **2 → 4** processes **increases throughput but scales poorly on CPU** (≈**25%** efficiency) because communication and synchronization dominate. At **P=2**, both **1F1B** and **Interleaved** maintain ~**50–60%**; at **P=4**, their scheduling advantages are largely eaten by overhead.

A forward-then-backward pipeline of **M** micro-batches costs ≈ `2M + 2(K−1)` ticks. **1F1B** overlaps the middle bubbles, reducing the fixed `O(K)` overhead; **Interleaved** shortens fill/drain by ~`1/v` with virtual stages. On CPU, the added inter-rank hand-offs at higher **P** can outweigh those savings.


## CPU impact:

Running pipeline schedules on **CPU** with the **gloo** backend shifts the balance between **compute** and **communication** compared to the typical **GPU + NCCL** setting:

- **Higher per-message latency (gloo on CPU).**  
  Inter-rank transfers and barriers are handled by gloo (often over loopback TCP on a single node). Small tensors and frequent sends/receives (especially with **Interleaved**, which creates **v×** more stage boundaries) become **latency dominated**, shrinking the benefit from bubble reduction.

- **Lower compute per stage ⇒ worse compute/comm ratio.**  
  On CPU, a Transformer layer is orders of magnitude slower than on GPU, but **communication and synchronization don’t shrink proportionally**. When each stage is “cheap,” the relative cost of coordination (barriers, sends, context switches) **dominates**. This is why **GPipe** (fewer transfers) can look better at larger **P** even though it has bigger bubbles.

- **Process scheduling & barriers.**  
  `torchrun` uses **multi-process** training. Each micro-batch step in these schedules introduces **barriers** and hand-offs. On CPU, OS context switches and Python control overhead are non-trivial; at **P=4** these costs stack up and **erode 1F1B/Interleaved gains**.

- **Backend & kernel efficiency.**  
  GPUs use **NCCL** and **fused kernels**; comm can overlap with compute and use GPUDirect/NVLink. On CPU, you lack those; you also pay more Python-side overhead per micro-batch/schedule step.

- **Memory hierarchy & NUMA effects.**  
  Multiple CPU ranks can thrash caches and contend for memory bandwidth. If ranks land on different NUMA nodes, cross-node traffic increases variability and latency (worsens with more ranks and more stage boundaries).

- **Per-microbatch overheads.**  
  Pipeline micro-batching reduces bubbles, but **each micro-batch introduces fixed costs** (autograd bookkeeping, dispatcher overhead, send/recv). With **small micro-batches** those costs are a large fraction of total time.

**Implication.**  
On CPU, increasing **P** raises tokens/s but **scales poorly** because comm/sync overheads grow faster than useful compute per stage. Hence the pattern you see: **1F1B** wins at small **P** (great overlap, limited comm), **Interleaved** helps at **P=2** for wider models (more virtual stages keep ranks busy), but at **P=4** the extra hand-offs flatten or reverse the gains; **GPipe**’s simplicity becomes competitive.



