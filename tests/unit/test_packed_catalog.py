"""Exact concurrent catalog preparation, resume pins, and bounded index reads."""

import io
import sqlite3

import numpy as np
import pytest
from test_packed_text import ReadOnlyR2, fixture

from teutonic.evaluation.packed_catalog import (
    build_resumable_catalog,
    merge_catalogs,
    partition_rows,
)
from teutonic.evaluation.packed_text import BUCKETS, DTYPE, DatasetReader, build_catalog, canonical


@pytest.mark.parametrize('page_bytes', [512,4096,65536])
def test_matches_sequential_catalog_and_resumes_without_r2(tmp_path,page_bytes):
    client,dataset,_,_=fixture()
    expected=canonical(build_catalog(dataset))
    client.calls.clear()
    state=tmp_path/'catalog.sqlite'
    actual=build_resumable_catalog(dataset,state,workers=3,page_bytes=page_bytes)
    assert canonical(actual)==expected
    assert client.calls and all(key.endswith('.documents.npy') for key,_ in client.calls)
    for _,value in client.calls:
        start,end=map(int,value.removeprefix('bytes=').split('-'))
        assert end-start+1<=page_bytes
    client.calls.clear()
    assert canonical(build_resumable_catalog(dataset,state,workers=1))==expected
    assert not client.calls


def test_resume_rejects_dataset_and_checkpoint_changes(tmp_path):
    _,dataset,_,_=fixture()
    state=tmp_path/'catalog.sqlite'
    build_resumable_catalog(dataset,state,workers=2)
    pin=dataset.pin
    dataset.pin={**pin,'sha256':'f'*64}
    with pytest.raises(ValueError,match='another dataset'):
        build_resumable_catalog(dataset,state)
    dataset.pin=pin
    with sqlite3.connect(state) as db:
        db.execute("UPDATE partitions SET rows='[]'")
    with pytest.raises(ValueError,match='checksum mismatch'):
        build_resumable_catalog(dataset,state)


def test_failed_partition_can_resume_completed_work(tmp_path,monkeypatch):
    _,dataset,_,_=fixture()
    original=dataset.reader.read
    def fail(obj,*args):
        if 'part-1' in obj['path']:
            raise RuntimeError('fixture read failure')
        return original(obj,*args)
    monkeypatch.setattr(dataset.reader,'read',fail)
    state=tmp_path/'catalog.sqlite'
    with pytest.raises(RuntimeError,match='fixture read failure'):
        build_resumable_catalog(dataset,state,workers=1)
    monkeypatch.setattr(dataset.reader,'read',original)
    assert canonical(build_resumable_catalog(dataset,state))==canonical(build_catalog(dataset))


def test_wrong_manifest_endpoint_is_rejected():
    _,dataset,_,_=fixture()
    obj=next(o for o in dataset.objects.values() if o['role']=='index')
    with pytest.raises(ValueError,match='endpoints differ'):
        partition_rows(dataset.reader,{**obj,'max_tokens':99999})


@pytest.mark.parametrize('page_bytes', [512, 4096, 65536])
def test_large_index_boundaries_match_full_array(page_bytes):
    lengths = np.sort(np.concatenate((
        np.random.default_rng(91).integers(0, 100000, 20000),
        np.repeat([1, 2, 4096, 4097, 16384, 16385], 75),
    )))
    array = np.zeros(len(lengths), dtype=DTYPE)
    array['original_mimo_token_count'] = lengths
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    client = ReadOnlyR2()
    obj = client.add('index/test/a/large.documents.npy', stream.getvalue(),
                     role='index', dataset='test', category='a', document_count=len(array),
                     min_tokens=int(lengths[0]), max_tokens=int(lengths[-1]))
    rows = partition_rows(DatasetReader(client), obj, page_bytes=page_bytes)
    assert [(r['bucket'], r['lo'], r['hi']) for r in rows] == [
        (name, int(np.searchsorted(lengths, low)), int(np.searchsorted(lengths, high, side='right')))
        for name, low, high, _ in BUCKETS
    ]


def test_sharded_merge_matches_original_and_rejects_partial_coverage(tmp_path):
    _,dataset,_,_=fixture()
    parts=[build_resumable_catalog(dataset,tmp_path/f'{i}.sqlite',workers=1,
                                    shard_index=i,shard_count=3) for i in range(3)]
    assert canonical(merge_catalogs(dataset,list(reversed(parts))))==canonical(build_catalog(dataset))
    with pytest.raises(ValueError,match='missing or duplicate'):
        merge_catalogs(dataset,parts[:-1])
    parts[0]['index_paths_sha256']='f'*64
    with pytest.raises(ValueError,match='identity/coverage'):
        merge_catalogs(dataset,parts)
