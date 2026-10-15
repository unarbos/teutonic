"""Bounded-memory NCCL replication of a model already loaded on one GPU."""

import hashlib
import json
import time
from datetime import timedelta

import torch
import torch.distributed as dist

BROADCAST_CHUNK_BYTES = 256 * 1024**2


def model_tensors(model):
    # named_buffers includes non-persistent buffers (e.g. rotary frequencies).
    # named_parameters deduplicates tied weights; receivers must restore the ties.
    return sorted([*model.named_parameters(), *model.named_buffers()], key=lambda item: item[0])


def tensor_chunks(items, chunk_bytes=BROADCAST_CHUNK_BYTES):
    """Yield same-dtype flat views, splitting even individual oversized tensors."""
    parts, count, dtype = [], 0, None
    for name, tensor in items:
        if not tensor.is_contiguous():
            raise ValueError(f"cannot broadcast non-contiguous model tensor: {name}")
        limit = chunk_bytes // tensor.element_size()
        if limit < 1:
            raise ValueError("broadcast chunk is smaller than one tensor element")
        for part in tensor.detach().view(-1).split(limit):
            if not part.numel():
                continue
            if parts and (part.dtype != dtype or count + part.numel() > limit):
                yield parts
                parts, count = [], 0
            dtype = part.dtype
            parts.append(part)
            count += part.numel()
    if parts:
        yield parts


def broadcast_model(model, *, rank, world_size, rendezvous, gpu_id,
                    chunk_bytes=BROADCAST_CHUNK_BYTES):
    """Copy all parameters/buffers from rank zero, then release the loading group.

    Call only after the parent confirms that every member has finished preparing.
    The parent must terminate all peers on failure; a failed collective cannot be
    recovered by continuing with a partially populated replica.
    """
    if dist.is_initialized():
        raise RuntimeError("model broadcast requires retiring the previous process group")
    device = torch.device("cuda", gpu_id)
    items = model_tensors(model)
    for name, tensor in items:
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"model tensor must be contiguous on {device}: {name}")
    signature = hashlib.sha256(json.dumps([
        (name, list(tensor.shape), str(tensor.dtype)) for name, tensor in items
    ]).encode()).hexdigest()
    started = time.monotonic()
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size,
        timeout=timedelta(minutes=3), device_id=device,
    )
    signatures = [None] * world_size
    dist.all_gather_object(signatures, signature)
    if len(set(signatures)) != 1:
        raise RuntimeError("model broadcast layouts differ across replicas")
    torch.cuda.synchronize(device)
    transfer_started = time.monotonic()
    chunks = 0
    with torch.inference_mode():
        for parts in tensor_chunks(items, chunk_bytes):
            flat = (torch.cat(parts) if rank == 0 else torch.empty(
                sum(part.numel() for part in parts), dtype=parts[0].dtype, device=device,
            ))
            dist.broadcast(flat, src=0)
            if rank != 0:
                offset = 0
                for part in parts:
                    part.copy_(flat[offset:offset + part.numel()])
                    offset += part.numel()
            del flat
            chunks += 1
        torch.cuda.synchronize(device)
    transfer_s = time.monotonic() - transfer_started
    # All ranks finish their copies before releasing the temporary communicator.
    dist.barrier(device_ids=[gpu_id])
    dist.destroy_process_group()
    return {
        "source_rank": 0, "rank": rank, "replicas": world_size,
        "bytes": sum(t.numel() * t.element_size() for _, t in items),
        "chunks": chunks, "transfer_s": transfer_s,
        "total_s": time.monotonic() - started,
    }
