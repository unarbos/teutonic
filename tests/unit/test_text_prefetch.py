import threading
from types import SimpleNamespace

import pytest
from test_packed_text import fixture

from teutonic.evaluation.packed_text import PackedDataset
from teutonic.evaluator.text_prefetch import prefetched_text_batches
from teutonic.evaluator.text_sampling import sample_documents


def test_fetches_next_batch_while_consumer_is_busy_and_bounds_lookahead(tmp_path):
    fetched, closed = [], []
    second_ready = threading.Event()

    def text(doc):
        fetched.append(doc.number)
        if doc.number == 1:
            second_ready.set()
        return "text"

    def batches():
        try:
            for i in range(100):
                yield [SimpleNamespace(number=i, byte_length=4)]
        finally:
            closed.append(True)

    source = batches()
    dataset = SimpleNamespace(text_cache=tmp_path, text=text)
    try:
        with (
            pytest.raises(RuntimeError, match="scorer failed"),
            prefetched_text_batches(dataset, source, workers=1) as ready,
        ):
            assert next(ready)[0].number == 0
            # No second next(): text arrives while the consumer scores batch 0.
            assert second_ready.wait(5)
            raise RuntimeError("scorer failed")
        assert fetched in ([0, 1], [0, 1, 2])
    finally:
        source.close()
    assert closed == [True]


def test_real_cache_preserves_batches_and_avoids_repeat_r2_reads(tmp_path):
    client, dataset, catalog, mix = fixture()
    dataset = PackedDataset(dataset.reader, dataset.pin, text_cache=tmp_path)
    docs, _ = sample_documents(dataset, catalog, mix, 123)
    expected = [docs[i:i + 2] for i in range(0, len(docs), 2)]
    client.calls.clear()
    with prefetched_text_batches(dataset, iter(expected)) as ready:
        actual = list(ready)
    assert actual == expected
    calls = list(client.calls)
    assert calls and all(call[0].endswith('.utf8') for call in calls)
    for doc in docs:
        assert dataset.text(doc)
    assert client.calls == calls  # Tokenization and MiMo reuse the staged text.


def test_read_error_propagates_and_all_reads_finish_before_cleanup(tmp_path):
    finished = threading.Event()

    def text(doc):
        try:
            raise ValueError("document text hash mismatch")
        finally:
            finished.set()

    dataset = SimpleNamespace(text_cache=tmp_path, text=text)
    with (
        pytest.raises(ValueError, match="hash mismatch"),
        prefetched_text_batches(dataset, iter([[SimpleNamespace(byte_length=4)]])) as ready,
    ):
        next(ready)
    assert finished.is_set()


def test_cache_and_byte_bound_required_before_fetch(tmp_path):
    calls = []
    dataset = SimpleNamespace(text_cache=None, text=lambda doc: calls.append(doc))
    with pytest.raises(ValueError, match="disk cache"), prefetched_text_batches(dataset, iter([])):
        pass
    dataset.text_cache = tmp_path
    with (
        pytest.raises(ValueError, match="byte bound"),
        prefetched_text_batches(dataset, iter([[SimpleNamespace(byte_length=5)]]),
                                max_document_bytes=4) as ready,
    ):
        next(ready)
    assert calls == []
