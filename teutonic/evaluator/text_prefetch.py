"""Stage selected UTF-8 batches on disk while the preceding batch is scored."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from teutonic.evaluator.text_sampling import prefetched_document_batches


@contextmanager
def prefetched_text_batches(dataset, batches, *, depth=2, workers=16,
                            max_document_bytes=64 * 1024 * 1024):
    """Keep only document descriptors in the lookahead queue, never corpus text.

    Text is verified by PackedDataset.text and placed in the duel's disk cache.
    The existing preparation/scoring path reads that cache without changing its
    tokenizer batches, reference-cache decisions, or inference request boundaries.
    Join outstanding reads before the caller removes the temporary cache.
    """
    if dataset.text_cache is None:
        raise ValueError("text prefetch requires a disk cache")
    if min(depth, workers, max_document_bytes) < 1:
        raise ValueError("text prefetch bounds must be positive")

    def fetch(doc):
        dataset.text(doc)  # Drop the decoded string rather than retaining it in a Future.

    def staged():
        for batch in batches:
            if any(doc.byte_length > max_document_bytes for doc in batch):
                raise ValueError("document exceeds configured byte bound; sample was not changed")
            # A fixed number of readers bounds live text memory. Completed futures
            # hold None; the much larger selected corpus lives only on disk.
            for _ in reads.map(fetch, batch):
                pass
            yield batch

    with (
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="text-prefetch") as reads,
        prefetched_document_batches(staged(), depth=depth) as ready,
    ):
        yield ready
