"""One trainer for every stage: torchrun --nproc_per_node N -m train.trainer configs/1_fold.json

A config lists tasks with sampling weights; each step accumulates `effective_batch` items, drawn task by task.
"""
import contextlib
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel as DDP

from rumifold import RumiFold


class Wrapped(torch.nn.Module):
    """DDP needs one forward: run one task's loss on one item."""

    def __init__(self, model, tasks):
        super().__init__(); self.model, self.tasks = model, tasks

    def forward(self, name, item): return self.tasks[name].loss(self.model, item)


def load_task(name, cfg):
    mod = importlib.import_module(f"train.tasks.{name}")
    return getattr(mod, "".join(x.title() for x in name.split("_")))(cfg)


def main(path):
    cfg = json.loads(Path(path).read_text())
    world, rank, local = (int(os.environ.get(k, d)) for k, d in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    torch.cuda.set_device(local)
    if world > 1: dist.init_process_group("nccl")
    if cfg["effective_batch"] % world: sys.exit("effective_batch must be a multiple of the number of GPUs")
    accumulate = cfg["effective_batch"] // world; out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    model = RumiFold.for_training(**cfg["model"])
    folder = model.fold_model
    adapter = folder.qwen.layers[0].self_attn.q_proj
    rank_lora = adapter.a.out_features
    inference_config = {
        "base_model": folder.qwen.config._name_or_path,
        "base_model_revision": getattr(folder.qwen.config, "_commit_hash", None),
        "lora": {"rank": rank_lora, "alpha": adapter.scale * rank_lora},
        "chain_gap": folder.gap,
    }
    tasks = {n: load_task(n, c) for n, c in cfg["tasks"].items()}; names = list(tasks); weights = [cfg["tasks"][n]["weight"] for n in names]
    for t in tasks.values():
        if hasattr(t, "setup"): t.setup(model)                      # supplementary heads
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    group = lambda n: next((v for k, v in cfg.get("lr_by_prefix", {}).items() if n.startswith(k)), cfg["lr"])
    by_lr = {}
    for n, p in params.items(): by_lr.setdefault(group(n), []).append(p)
    opt = torch.optim.AdamW([dict(params=ps, lr=lr, base_lr=lr) for lr, ps in by_lr.items()], weight_decay=cfg["weight_decay"])
    start, rng = 0, random.Random(cfg["seed"] + 100003 * rank)
    resume = sorted(out.glob("step-*.pt"))
    if resume:
        s = torch.load(resume[-1], map_location="cpu", weights_only=True); start = s["step"]
        model.load_state_dict(s["model"], strict=False); opt.load_state_dict(s["optimizer"]); rng.seed(cfg["seed"] + 100003 * rank + start)
    torch.manual_seed(cfg["seed"] + 100003 * rank + start)
    net = Wrapped(model, tasks)
    if world > 1: net = DDP(net, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True)
    begin = time.monotonic()
    for step in range(start + 1, cfg["steps"] + 1):
        for g in opt.param_groups: g["lr"] = g["base_lr"] * min(1., step / cfg["warmup"])
        opt.zero_grad(set_to_none=True); total = 0.
        for micro in range(accumulate):
            name = rng.choices(names, weights)[0]
            item = {k: v.cuda() if torch.is_tensor(v) else v for k, v in tasks[name].draw(rng, step).items()}
            hold = net.no_sync() if world > 1 and micro < accumulate - 1 else contextlib.nullcontext()
            with hold:
                loss, parts = net(name, item)
                (loss / accumulate).backward(); total += float(loss.detach()) / accumulate
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True); opt.step()
        if rank == 0:
            with (out / "log.jsonl").open("a") as f: f.write(json.dumps(dict(step=step, loss=total, last_task=name, parts=parts, grad_norm=float(norm), lr=opt.param_groups[0]["lr"], seconds=time.monotonic() - begin)) + "\n")
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            if rank == 0:
                config_tmp = out / "config.json.tmp"
                config_tmp.write_text(json.dumps(inference_config, indent=2) + "\n")
                config_tmp.replace(out / "config.json")
                state = {n: p.detach().cpu() for n, p in params.items()}
                tmp = out / f"step-{step:07d}.tmp"; torch.save(dict(step=step, model=state, optimizer=opt.state_dict()), tmp); tmp.rename(tmp.with_suffix(".pt"))
                save_file({k.replace("fold_model.", "fold.", 1): v.contiguous() for k, v in state.items()}, str(out / "model.safetensors"))
            if world > 1: dist.barrier()
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__": main(sys.argv[1])
