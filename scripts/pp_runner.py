import os, time, argparse
import torch, torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass
from torch.distributed.pipelining import (
    PipelineStage, ScheduleGPipe, Schedule1F1B, ScheduleInterleaved1F1B,
    pipeline, SplitPoint
)

@dataclass
class ModelArgs:
    dim: int = 480
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 10000

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleDict({str(i): nn.TransformerDecoderLayer(args.dim, args.n_heads) for i in range(args.n_layers)})
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size)

    def forward(self, tokens: torch.Tensor):
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        for layer in self.layers.values():
            h = layer(h, h)
        h = self.norm(h) if self.norm else h
        return self.output(h).clone() if self.output else h

def tokenwise_ce(outputs, targets, vocab_size):
    loss = nn.CrossEntropyLoss()
    return loss(outputs.reshape(-1, vocab_size), targets.reshape(-1))

def init_distributed(backend="gloo"):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=backend)
    return rank, world

def manual_model_split_n(model: Transformer, rank: int, world: int, device: torch.device):
    n = model.args.n_layers
    if world > n:
        raise ValueError("world_size cannot exceed n_layers in this demo.")
    start = (rank * n) // world
    end = ((rank + 1) * n) // world
    keep = set(str(i) for i in range(start, end))
    for i in range(n):
        if str(i) not in keep:
            del model.layers[str(i)]
    if rank != 0:
        model.tok_embeddings = None
    if rank != world - 1:
        model.norm = None
        model.output = None
    return PipelineStage(model, rank, world, device)

def tracer_split_every_layer(model, example_input_microbatch, rank, world, device, group=None):
    n_layers = model.args.n_layers
    split_spec = {f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)}
    pipe = pipeline(module=model, mb_args=(example_input_microbatch,), split_spec=split_spec)
    num_stages = n_layers
    local_stage_ids = [i for i in range(num_stages) if (i % world) == rank]
    pg = group if group is not None else dist.group.WORLD
    local_stage_modules = [pipe.get_stage_module(i) for i in local_stage_ids]
    local_stage_runtimes = [pipe.build_stage(i, device, pg) for i in local_stage_ids]
    last_rank = (num_stages - 1) % world
    return local_stage_modules, local_stage_runtimes, num_stages, last_rank

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=["gpipe","1f1b","interleaved"], default="gpipe")
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=480)
    ap.add_argument("--vocab-size", type=int, default=10000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--microbatches", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=1)
    ap.add_argument("--measure-steps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--emit-csv", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    rank, world = init_distributed("gloo")
    device = torch.device("cpu")
    first_rank = 0

    model = Transformer(ModelArgs(args.dim, args.n_layers, args.n_heads, args.vocab_size))
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), dtype=torch.long)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), dtype=torch.long)
    example_input_microbatch = x.chunk(args.microbatches)[0]

    def loss_fn(o, t):
        return tokenwise_ce(o, t, args.vocab_size)

    if args.schedule == "interleaved":
        local_mods, local_stages, num_stages, last_rank = tracer_split_every_layer(model, example_input_microbatch, rank, world, device)
        local_params = []
        for m in local_mods:
            local_params += list(m.parameters())
        opt = torch.optim.Adam(local_params, lr=3e-4)
        schedule = ScheduleInterleaved1F1B(local_stages, n_microbatches=args.microbatches, loss_fn=loss_fn)

        def one_step():
            opt.zero_grad(set_to_none=True)
            if rank == first_rank:
                schedule.step(x.to(device))
            elif rank == last_rank:
                losses = []
                _ = schedule.step(target=y.to(device), losses=losses)
            else:
                schedule.step()
            dist.barrier()
            opt.step()
    else:
        stage = manual_model_split_n(model, rank, world, device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        if args.schedule == "gpipe":
            schedule = ScheduleGPipe(stage, n_microbatches=args.microbatches, loss_fn=loss_fn)
        elif args.schedule == "1f1b":
            schedule = Schedule1F1B(stage, n_microbatches=args.microbatches, loss_fn=loss_fn)
        else:
            raise ValueError("unknown schedule")
        last_rank = world - 1

        def one_step():
            opt.zero_grad(set_to_none=True)
            if rank == first_rank:
                _ = schedule.step(x.to(device))
            elif rank == last_rank:
                losses = []
                _ = schedule.step(target=y.to(device), losses=losses)
            else:
                schedule.step()
            dist.barrier()
            opt.step()

    for _ in range(args.warmup_steps):
        one_step()

    start = time.perf_counter()
    for _ in range(args.measure_steps):
        one_step()
    avg_elapsed = (time.perf_counter() - start) / args.measure_steps
    total_elapsed = avg_elapsed
