"""Two-replica full-context scoring for documents exceeding the tested GPU limit."""

import math
import tempfile
from datetime import timedelta
from pathlib import Path
from queue import Empty

import torch
import torch.distributed as dist
import torch.nn.functional as F

from teutonic.evaluation.masking import DOCUMENT_EOS_TOKEN_ID

SINGLE_GPU_MAX_TOKENS = 128_974


def worker_pairs(specs):
    pairs = []
    for role in ("king", "challenger"):
        workers = [s["worker_id"] for s in specs if s["role"] == role]
        if len(workers) < 2 or len(workers) % 2:
            raise ValueError("sequence parallelism requires pairs of replicas within each model role")
        pairs.extend({"role": role, "workers": workers[i:i + 2]} for i in range(0, len(workers), 2))
    return pairs


def initialize(rank, rendezvous, gpu_id):
    if dist.is_initialized():
        raise RuntimeError("sequence-parallel process group is already initialized")
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(minutes=3), device_id=torch.device("cuda", gpu_id),
    )


def split_lengths(length):
    width = (length + 1) // 2
    return [width, length - width]


def receive(pool, generation, on_progress):
    while True:
        try:
            message = pool.result_queue.get(timeout=30)
        except Empty:
            dead = pool.dead_workers()
            if dead:
                raise RuntimeError(f"workers exited during sequence-parallel scoring: {dead}") from None
            on_progress({"phase": "sequence_parallel_wait"})
            continue
        if message["type"] == "error":
            raise RuntimeError(f"sequence-parallel worker {message['worker_id']} failed: {message['error']}\n{message.get('traceback', '')}")
        if message.get("generation") != generation:
            raise RuntimeError("stale sequence-parallel worker result")
        return message


def prepare_workers(pool, generation, on_progress):
    if getattr(pool, "parallel_pairs", None) is not None:
        return pool.parallel_pairs
    pairs = worker_pairs(pool.specs)
    pool.parallel_directory = tempfile.TemporaryDirectory(prefix="teutonic-sequence-parallel-")
    expected = {}
    on_progress({"phase": "sequence_parallel_init", "groups": len(pairs)})
    for index, pair in enumerate(pairs):
        rendezvous = (Path(pool.parallel_directory.name) / f"pair-{index}").as_uri()
        for rank, worker_id in enumerate(pair["workers"]):
            expected[worker_id] = rank
            pool.command_queues[worker_id].put({"type": "sequence_parallel_init", "generation": generation,
                                               "rank": rank, "rendezvous": rendezvous})
    while expected:
        message = receive(pool, generation, on_progress)
        worker = message["worker_id"]
        if message["type"] != "sequence_parallel_ready" or worker not in expected or message.get("rank") != expected[worker]:
            raise RuntimeError("invalid sequence-parallel initialization result")
        pool.ready[worker]["sequence_parallel_rank"] = expected.pop(worker)
    pool.parallel_pairs = pairs
    return pairs


def submit(pool, workers, sequences, index, generation):
    for worker in workers:
        pool.command_queues[worker].put({"type": "score_parallel", "generation": generation,
                                       "sequence_indices": [index], "token_batches": [sequences[index]]})


def result_loss(message, index):
    if (message["type"] != "result" or message.get("sequence_indices") != [index]
            or len(message.get("losses", [])) != 1 or message.get("scoring_mode") != "sequence_parallel"):
        raise RuntimeError("invalid sequence-parallel scoring result")
    value = float(message["losses"][0])
    if not math.isfinite(value):
        raise RuntimeError("non-finite sequence-parallel loss")
    return value


def score_documents(pool, sequences, indices, generation, on_progress):
    """Finish large documents on replica pairs before ordinary batching resumes."""
    for index in indices:
        row = sequences[index]
        if len(row) <= SINGLE_GPU_MAX_TOKENS or row[-1] != DOCUMENT_EOS_TOKEN_ID or DOCUMENT_EOS_TOKEN_ID in row[:-1]:
            raise ValueError("sequence-parallel input must be a complete document above the single-GPU cutoff")
    pairs = prepare_workers(pool, generation, on_progress)
    longest = max(indices, key=lambda i: len(sequences[i]))
    pending = {worker for pair in pairs for worker in pair["workers"]}
    on_progress({"phase": "long_document_preflight_start", "sequence_index": longest,
                 "input_tokens": len(sequences[longest]), "total_workers": len(pending),
                 "scoring_mode": "sequence_parallel"})
    for pair in pairs:
        submit(pool, pair["workers"], sequences, longest, generation)
    preflight_results = []
    while pending:
        message = receive(pool, generation, on_progress)
        if message["worker_id"] not in pending:
            raise RuntimeError("unexpected sequence-parallel preflight worker")
        result_loss(message, longest)
        pending.remove(message["worker_id"])
        preflight_results.append({k: message.get(k) for k in (
            "worker_id", "role", "wall_time_s", "peak_allocated_gib", "oom_retries", "scoring_mode",
        )})
        on_progress({"phase": "long_document_preflight_passed", "worker_id": message["worker_id"],
                     "input_tokens": len(sequences[longest]), "remaining_workers": len(pending),
                     "scoring_mode": "sequence_parallel"})

    positions = {"king": 0, "challenger": 0}
    losses = {"king": {}, "challenger": {}}
    active = {}
    worker_pair = {worker: i for i, pair in enumerate(pairs) for worker in pair["workers"]}
    records = []

    def dispatch(pair_id):
        pair = pairs[pair_id]
        role = pair["role"]
        if positions[role] < len(indices):
            index = indices[positions[role]]
            positions[role] += 1
            active[pair_id] = {"index": index, "results": {}}
            submit(pool, pair["workers"], sequences, index, generation)

    for pair_id in range(len(pairs)):
        dispatch(pair_id)
    while active:
        message = receive(pool, generation, on_progress)
        worker = message["worker_id"]
        pair_id = worker_pair.get(worker)
        if pair_id not in active or worker in active[pair_id]["results"]:
            raise RuntimeError("unexpected or duplicate sequence-parallel result")
        job, pair = active[pair_id], pairs[pair_id]
        if message.get("role") != pair["role"]:
            raise RuntimeError("sequence-parallel result has the wrong model role")
        job["results"][worker] = result_loss(message, job["index"])
        records.append({"sequence_index": job["index"], **{k: message.get(k) for k in (
            "worker_id", "role", "wall_time_s", "peak_allocated_gib", "scoring_mode",
        )}})
        if len(job["results"]) == 2:
            values = list(job["results"].values())
            if values[0] != values[1]:
                raise RuntimeError("sequence-parallel ranks disagree on the document loss")
            losses[pair["role"]][job["index"]] = values[0]
            del active[pair_id]
            on_progress({"phase": "sequence_parallel_progress", "role": pair["role"],
                         "done": len(losses[pair["role"]]), "total": len(indices)})
            dispatch(pair_id)
    if any(set(values) != set(indices) for values in losses.values()):
        raise RuntimeError("sequence-parallel scoring did not cover every oversized document")
    return losses, {
        "single_gpu_max_tokens": SINGLE_GPU_MAX_TOKENS, "gpus_per_document": 2,
        "sequence_indices": indices, "pairs": pairs, "results": records,
        "preflight": {"sequence_index": longest, "input_tokens": len(sequences[longest]),
                      "scoring_mode": "sequence_parallel", "workers": preflight_results},
    }


class PrefixAttention:
    def __init__(self, backend, rank, lengths):
        self.backend, self.rank, self.lengths = backend, rank, lengths
        self.layers = set()

    def gather(self, states):
        width = max(self.lengths)
        padded = F.pad(states, (0, 0, 0, width - states.shape[2])).contiguous()
        received = [torch.empty_like(padded) for _ in self.lengths]
        dist.all_gather(received, padded)
        return torch.cat(
            [received[i][:, :, :self.lengths[i]] for i in range(self.rank + 1)], dim=2,
        )

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        local_length = self.lengths[self.rank]
        if attention_mask is not None or any(t.shape[2] != local_length for t in (query, key, value)):
            raise RuntimeError("invalid sequence-parallel attention inputs")
        global_key, global_value = self.gather(key), self.gather(value)
        prefix_length = sum(self.lengths[:self.rank + 1])
        # FA4's lower-right causal alignment offsets the local queries by
        # prefix_length - local_length, exactly their global document position.
        kwargs.update(
            cu_seq_lens_q=torch.tensor([0, local_length], device=query.device, dtype=torch.int32),
            cu_seq_lens_k=torch.tensor([0, prefix_length], device=query.device, dtype=torch.int32),
            max_length_q=local_length, max_length_k=prefix_length,
        )
        self.layers.add(module.layer_idx)
        return self.backend(module, query, global_key, global_value, None, **kwargs)


@torch.no_grad()
def compute_loss(model, tokens, chunk_size, rank):
    """Score one entire document; both ranks return the same global mean loss."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if model.config.model_type != "mimo_v2" or model.config._attn_implementation != "flash_attention_4":
        raise ValueError("sequence-parallel scoring requires the pinned MiMo FA4 backend")
    if len(tokens) <= SINGLE_GPU_MAX_TOKENS:
        raise ValueError("sequence parallelism is reserved for documents over the single-GPU cutoff")
    if tokens[-1] != DOCUMENT_EOS_TOKEN_ID or DOCUMENT_EOS_TOKEN_ID in tokens[:-1]:
        raise ValueError("sequence-parallel input must be one complete document with terminal EOS")
    if not dist.is_initialized() or dist.get_world_size() != 2 or dist.get_rank() != rank:
        raise RuntimeError("sequence-parallel rank is not initialized correctly")
    lengths = split_lengths(len(tokens))
    start, end = sum(lengths[:rank]), sum(lengths[:rank + 1])
    device = model.model.embed_tokens.weight.device
    previous = ALL_ATTENTION_FUNCTIONS["flash_attention_4"]
    backend = getattr(previous, "_teutonic_document_backend", previous)
    adapter = PrefixAttention(backend, rank, lengths)
    ids = torch.tensor([tokens[start:end]], device=device)
    positions = torch.arange(start, end, device=device).unsqueeze(0)
    try:
        ALL_ATTENTION_FUNCTIONS.register("flash_attention_4", adapter)
        if hasattr(model, "reset_state"):
            model.reset_state()
        hidden = model.model(
            ids, position_ids=positions, use_cache=False,
            attention_mask={"full_attention": None, "sliding_window_attention": None},
        ).last_hidden_state
        if len(adapter.layers) != model.config.num_hidden_layers:
            raise RuntimeError("not every model layer used sequence-parallel attention")
    finally:
        ALL_ATTENTION_FUNCTIONS.register("flash_attention_4", previous)

    # Score the prediction crossing the GPU boundary. Only the document's last
    # hidden state has no next-token target; the second rank's first token counts.
    count = min(end, len(tokens) - 1) - start
    labels = torch.tensor(tokens[start + 1:start + count + 1], device=device)
    parts = []
    for offset in range(0, count, chunk_size):
        stop = min(offset + chunk_size, count)
        logits = model.lm_head(hidden[:, offset:stop])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[offset:stop], reduction="none")
        parts.append(loss.float())
        del logits
    local_losses = torch.cat(parts)
    counts = [lengths[0], lengths[1] - 1]
    padded = F.pad(local_losses, (0, max(counts) - count))
    received = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(received, padded)
    # Same full-vector float32 reduction as single-GPU scoring, rather than
    # averaging two rounded rank means with unequal target counts.
    losses = torch.cat([part[:n] for part, n in zip(received, counts, strict=True)])
    return (losses.sum() / (len(tokens) - 1)).item()
