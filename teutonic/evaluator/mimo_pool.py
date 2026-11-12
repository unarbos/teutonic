"""Four replicas per side, with two-GPU groups only for large documents."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from teutonic.evaluator.vllm_runtime import Worker, wait_for_gpu_release
from teutonic.evaluator.vllm_score import SINGLE_GPU_MAX_TOKENS


def balanced_groups(indices, sequences, count):
    """Deterministically balance prompt tokens, then restore input order per group."""
    groups, loads = [[] for _ in range(count)], [0] * count
    for index in sorted(indices, key=lambda i: (-len(sequences[i]), i)):
        target = min(range(count), key=lambda i: (loads[i], i))
        groups[target].append(index)
        loads[target] += len(sequences[index])
    return [sorted(group) for group in groups]


class MiMoPool:
    def __init__(self, king, challenger, gpu_ids, directory, identity, on_phase):
        if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
            raise ValueError("MiMo replica pool needs eight distinct GPUs")
        self.snapshots = {"king": king, "challenger": challenger}
        self.gpu_ids, self.directory = list(gpu_ids), directory
        self.identity, self.on_phase = identity, on_phase
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.engines, self.reports = {}, []
        self.tp, self.context, self.generation = None, None, 0

    def _release(self):
        engines, self.engines = self.engines, {}
        for worker in engines.values():
            try:
                worker.close()
            finally:
                self.reports.append(dict(worker.report))
        if engines:
            wait_for_gpu_release(self.gpu_ids)

    def _ensure(self, tp, context):
        if self.engines and self.tp == tp and self.context >= context:
            return
        self._release()
        self.tp, self.context = tp, context
        self.generation += 1
        self.on_phase({"phase": "mimo_replica_layout", "gpus_per_instance": tp,
                       "instances_per_side": 4 // tp, "max_model_len": context})
        try:
            for role, offset in (("king", 0), ("challenger", 4)):
                for replica in range(4 // tp):
                    ids = self.gpu_ids[offset + replica * tp:offset + (replica + 1) * tp]
                    self.engines[role, replica] = Worker(
                        role, self.snapshots[role], ids, self.directory, self.identity, self.on_phase,
                        worker_id=f"{role}-tp{tp}-{replica}-generation{self.generation}",
                        max_model_len=context,
                    )
            futures = [self.executor.submit(w.wait_ready) for w in self.engines.values()]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            self._release()
            raise

    def score(self, documents, sequences):
        if len(documents) != len(sequences) or not sequences:
            raise ValueError("unaligned or empty replica batch")
        results = {"king": [None] * len(documents), "challenger": [None] * len(documents)}
        # Large documents never cause ordinary documents to run on a different topology.
        for tp in (1, 2):
            indices = [i for i, tokens in enumerate(sequences)
                       if (len(tokens) <= SINGLE_GPU_MAX_TOKENS) == (tp == 1)]
            if not indices:
                continue
            context = SINGLE_GPU_MAX_TOKENS + 1 if tp == 1 else max(len(sequences[i]) for i in indices) + 1
            self._ensure(tp, context)
            groups = balanced_groups(indices, sequences, 4 // tp)
            futures = {}
            try:
                for (role, replica), worker in self.engines.items():
                    group = groups[replica]
                    if group:
                        future = self.executor.submit(worker.score, [documents[i] for i in group],
                                                      [sequences[i] for i in group])
                        futures[future] = role, group
                for future in as_completed(futures):
                    role, group = futures[future]
                    for index, row in zip(group, future.result(), strict=True):
                        if row["id"] != documents[index].doc_id:
                            raise ValueError("replica response order/identity differs")
                        results[role][index] = row
            except BaseException:
                self._release()
                raise
        if any(row is None for rows in results.values() for row in rows):
            raise ValueError("replica pool omitted a document")
        return results["king"], results["challenger"]

    def close(self):
        try:
            self._release()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
