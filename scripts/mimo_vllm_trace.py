"""Opt-in, test-only first-token probes for the isolated MiMo vLLM adapter."""

import hashlib
import json
import os
from pathlib import Path


def install_trace(model):
    import torch
    import torch.distributed as dist

    root = Path(os.environ["TEUTONIC_MIMO_TRACE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    stream = (root / f"rank-{rank}.jsonl").open("w", buffering=1)
    reference = {}
    state = {"step": 0, "indices": [], "tokens": [], "size": 0}

    def begin(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        state["step"] += 1
        state["indices"], state["tokens"] = [], []
        if ids is None or positions is None:
            return
        state["size"] = ids.shape[0]
        candidates = ((positions == 0) & ((ids == 1474) | (ids == 785))).nonzero().flatten()
        for index in candidates.tolist():
            token = int(ids[index])
            if token not in state["tokens"]:
                state["indices"].append(index)
                state["tokens"].append(token)

    def record(name, phase, values):
        if not state["indices"]:
            return
        if isinstance(values, dict):
            values = tuple(values.values())
        if not isinstance(values, (tuple, list)):
            values = (values,)
        for slot, tensor in enumerate(values):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tensor.shape[0] != state["size"]:
                continue
            for index, token in zip(state["indices"], state["tokens"], strict=True):
                value = tensor[index].detach().float().cpu().clone()
                key = (name, phase, slot, token)
                digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
                old = reference.setdefault(key, value)
                delta = (value - old).abs()
                row = {"step": state["step"], "name": name, "phase": phase,
                       "slot": slot, "token": token, "row": index,
                       "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                       "hash": digest, "different": int((delta != 0).sum()),
                       "max_abs": float(delta.max())}
                stream.write(json.dumps(row) + "\n")

    # vLLM's support_torch_compile __call__ bypasses nn.Module pre-hooks,
    # including when enforce_eager=True. Wrap forward explicitly instead.
    original_forward = model.model.forward

    def traced_forward(*args, **kwargs):
        begin(model.model, args, kwargs)
        return original_forward(*args, **kwargs)

    model.model.forward = traced_forward
    for name, module in model.model.named_modules():
        if name == "embed_tokens" or name.startswith(("layers.0.", "layers.1.")) or name in ("layers.0", "layers.1"):
            def before(module, args, kwargs, name=name):
                record(name, "in", (*args, *kwargs.values()))

            def after(module, args, kwargs, output, name=name):
                record(name, "out", output)

            module.register_forward_pre_hook(before, with_kwargs=True)
            module.register_forward_hook(after, with_kwargs=True)
