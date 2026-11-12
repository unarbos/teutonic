"""Bounded, ordered text preparation. Workers cannot change sample membership."""

import json
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class TextTokenizer:
    """One encoding per document; correctness comparisons live in verification tests.

    Append the model's EOS, with no BOS; score targets 1..N-1 for both models.
    This preserves one-token documents in the original 2-token minimum bucket.
    """

    def __init__(self, snapshot, *, backend="gigatoken", eos_token_id=None):
        os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("TEUTONIC_TOKENIZER_THREADS", "8"))
        from tokenizers import Tokenizer

        self.reference = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))
        self.reference.no_truncation()
        self.reference.no_padding()
        if eos_token_id is None:
            eos_token_id = json.loads((Path(snapshot) / "config.json").read_text())["eos_token_id"]
        if type(eos_token_id) is not int or not 0 <= eos_token_id < self.reference.get_vocab_size():
            raise ValueError("one valid EOS token ID is required by the text scoring policy")
        self.eos_token_id = eos_token_id
        self.fast = None
        self.fast_documents = 0
        self.reference_documents = 0
        self.lock = threading.Lock()
        if backend == "gigatoken":
            import gigatoken
            self.fast = gigatoken.Tokenizer(self.reference)
        elif backend != "reference":
            raise ValueError("unknown tokenizer backend")

    def encode(self, texts):
        with self.lock:
            return self._encode(texts)

    def _encode(self, texts):
        if self.fast is not None:
            encoded = self.fast.encode_batch_list(texts)
            self.fast_documents += len(texts)
        else:
            encoded = [x.ids for x in self.reference.encode_batch(texts, add_special_tokens=False)]
            self.reference_documents += len(texts)
        return [row + [self.eos_token_id] for row in encoded]

    def status(self):
        return {"backend": "gigatoken" if self.fast is not None else "reference",
                "gigatoken_documents": self.fast_documents,
                "reference_documents": self.reference_documents,
                "reference_comparison": "verification-tests-only"}


def prepared_batches(dataset, documents, tokenizer, *, batch_documents=32,
                     prefetch_batches=2, max_batch_bytes=8 * 1024 * 1024,
                     max_document_bytes=64 * 1024 * 1024, max_tokens=16384, workers=8):
    """Bound both in-flight documents and text bytes; yield in selected order."""
    if min(batch_documents, prefetch_batches, max_batch_bytes, max_document_bytes, workers) < 1:
        raise ValueError("preparation bounds must be positive")
    def groups():
        group, size = [], 0
        for doc in documents:
            if doc.byte_length > max_document_bytes:
                raise ValueError("document exceeds configured byte bound; sample was not changed")
            if group and (len(group) >= batch_documents or size + doc.byte_length > max_batch_bytes):
                yield group
                group, size = [], 0
            group.append(doc)
            size += doc.byte_length
        if group:
            yield group

    def prepare(group):
        # The outer bounded executor overlaps range reads and tokenization with scoring.
        texts = list(reads.map(dataset.text, group))
        tokens = tokenizer.encode(texts)
        if len(tokens) != len(group) or any(not 2 <= len(t) <= max_tokens for t in tokens):
            raise ValueError("tokenized document outside context policy; no truncation or replacement")
        return group, tokens

    with ThreadPoolExecutor(max_workers=workers) as reads, ThreadPoolExecutor(
        max_workers=min(workers, prefetch_batches)
    ) as pool:
        iterator, pending = iter(groups()), deque()
        try:
            for _ in range(prefetch_batches):
                group = next(iterator, None)
                if group is not None:
                    pending.append(pool.submit(prepare, group))
            while pending:
                batch = pending.popleft().result()
                group = next(iterator, None)
                if group is not None:
                    pending.append(pool.submit(prepare, group))
                yield batch
        finally:
            for future in pending:
                future.cancel()
