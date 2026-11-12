"""Resumable, bounded preparation of exact packed-text length-bucket catalogs."""

import fcntl
import hashlib
import io
import json
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from pathlib import Path

import numpy as np

from teutonic.evaluation.packed_text import BUCKETS, CATALOG_FORMAT, DTYPE, canonical, digest


def partition_rows(reader, obj, *, page_bytes=4096):
    """Binary-search exact row boundaries using small, conditional byte ranges."""
    if page_bytes < 512:
        raise ValueError('catalog pages must hold an NPY header')
    cache = {}
    def page(number):
        if number not in cache:
            start = number * page_bytes
            cache[number] = reader.read(obj, start, min(page_bytes, obj['size_bytes']-start))
        return cache[number]
    header = io.BytesIO(page(0))
    version = np.lib.format.read_magic(header)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(header)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(header)
    else:
        raise ValueError('unsupported NPY version')
    offset = header.tell()
    if (dtype != DTYPE or fortran or len(shape) != 1 or shape[0] != obj['document_count']
            or offset + shape[0]*DTYPE.itemsize != obj['size_bytes'] or not shape[0]):
        raise ValueError('index layout differs from pinned manifest')
    count = shape[0]
    field = DTYPE.fields['original_mimo_token_count'][1]
    def length(number):
        start = offset + number*DTYPE.itemsize + field
        first, last = start//page_bytes, (start+7)//page_bytes
        data = b''.join(page(i) for i in range(first,last+1))
        return int.from_bytes(data[start%page_bytes:start%page_bytes+8], 'little', signed=True)
    if length(0) != obj['min_tokens'] or length(count-1) != obj['max_tokens']:
        raise ValueError('index endpoints differ from pinned manifest')
    cuts = {}
    previous = 0
    for boundary in sorted({b[1] for b in BUCKETS} | {b[2]+1 for b in BUCKETS}):
        if obj['min_tokens'] >= boundary:
            found = 0
        elif obj['max_tokens'] < boundary:
            found = count
        else:
            lo, hi = previous, count
            while lo < hi:
                mid = (lo+hi)//2
                if length(mid) < boundary:
                    lo = mid+1
                else:
                    hi = mid
            found = lo
        if ((found and length(found-1) >= boundary)
                or (found < count and length(found) < boundary)):
            raise ValueError('bucket boundary fence failed')
        cuts[boundary] = previous = found
    return [{'dataset':obj['dataset'], 'category':obj['category'], 'bucket':name,
             'path':obj['path'], 'lo':cuts[minimum], 'hi':cuts[maximum+1]}
            for name,minimum,maximum,_ in BUCKETS if cuts[maximum+1] > cuts[minimum]]


def build_resumable_catalog(dataset, state_path, *, workers=64, page_bytes=4096,
                            on_progress=lambda event: None, shard_index=0, shard_count=1):
    if workers < 1 or workers > 512:
        raise ValueError('catalog workers must be in 1..512')
    if not 0 <= shard_index < shard_count:
        raise ValueError('invalid catalog shard')
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _build(dataset,path,workers,page_bytes,on_progress,shard_index,shard_count)


def _build(dataset,path,workers,page_bytes,on_progress,shard_index,shard_count):
    fingerprint = digest({'dataset':dataset.pin,'buckets':BUCKETS,'algorithm':'exact-boundaries-v1',
                          'shard':[shard_index,shard_count]})
    ordered = sorted((key,value) for key,value in dataset.objects.items() if value['role']=='index')
    indexes = {key:value for number,(key,value) in enumerate(ordered) if number % shard_count == shard_index}
    del ordered
    started = last = time.monotonic()
    completed = set()
    with closing(sqlite3.connect(path)) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS metadata (fingerprint TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS partitions (path TEXT PRIMARY KEY, identity TEXT NOT NULL, rows TEXT NOT NULL, sha256 TEXT NOT NULL)')
        pin = db.execute('SELECT fingerprint FROM metadata').fetchall()
        if pin and pin != [(fingerprint,)]:
            raise ValueError('catalog checkpoint belongs to another dataset or policy')
        if not pin:
            db.execute('INSERT INTO metadata VALUES (?)',(fingerprint,))
            db.commit()
        for key,identity,raw,sha in db.execute('SELECT path,identity,rows,sha256 FROM partitions'):
            if (key not in indexes or identity != digest(indexes[key])
                    or hashlib.sha256(raw.encode()).hexdigest()!=sha):
                raise ValueError('catalog checkpoint identity/checksum mismatch')
            completed.add(key)
        resumed = len(completed)
        iterator = iter(obj for key,obj in indexes.items() if key not in completed)
        on_progress({'completed':resumed,'total':len(indexes),'resumed':resumed,'workers':workers})
        pool = ThreadPoolExecutor(max_workers=workers)
        pending = {}
        try:
            def fill():
                while len(pending) < workers*2:
                    obj = next(iterator,None)
                    if obj is None:
                        return
                    future = pool.submit(partition_rows,dataset.reader,obj,page_bytes=page_bytes)
                    pending[future]=obj
            fill()
            while pending:
                ready,_ = wait(pending,return_when=FIRST_COMPLETED)
                for future in ready:
                    obj=pending.pop(future)
                    raw=canonical(future.result()).decode()
                    db.execute('INSERT INTO partitions VALUES (?,?,?,?)',
                               (obj['path'],digest(obj),raw,hashlib.sha256(raw.encode()).hexdigest()))
                    completed.add(obj['path'])
                now=time.monotonic()
                if len(completed)%500 < len(ready) or now-last>=10:
                    db.commit()
                    elapsed=now-started
                    on_progress({'completed':len(completed),'total':len(indexes),'resumed':resumed,
                                 'elapsed_seconds':elapsed,'partitions_per_second':(len(completed)-resumed)/elapsed})
                    last=now
                fill()
        finally:
            db.commit()
            pool.shutdown(wait=True,cancel_futures=True)
        if len(completed)!=len(indexes):
            raise ValueError('incomplete catalog checkpoint')
        rows=[]
        for (raw,) in db.execute('SELECT rows FROM partitions ORDER BY path'):
            rows.extend(json.loads(raw))
    on_progress({'completed':len(completed),'total':len(indexes),'resumed':resumed,
                 'elapsed_seconds':time.monotonic()-started,'state':'complete'})
    result = {'format':CATALOG_FORMAT,'dataset':dataset.pin,
              'buckets':[list(x) for x in BUCKETS],'partitions':rows}
    if shard_count > 1:
        result.update(format=CATALOG_FORMAT+'-partial',shard=[shard_index,shard_count],
                      index_partitions=len(indexes),index_paths_sha256=digest(sorted(indexes)))
    return result


def merge_catalogs(dataset, parts):
    count = len(parts)
    if not count or {tuple(p['shard']) for p in parts} != {(i,count) for i in range(count)}:
        raise ValueError('missing or duplicate catalog shards')
    paths = sorted(key for key,obj in dataset.objects.items() if obj['role']=='index')
    rows=[]
    for part in parts:
        index,total=part['shard']
        selected=paths[index::total]
        if (part['format']!=CATALOG_FORMAT+'-partial' or part['dataset']!=dataset.pin
                or part['buckets']!=[list(x) for x in BUCKETS]
                or part['index_partitions']!=len(selected)
                or part['index_paths_sha256']!=digest(selected)):
            raise ValueError('catalog shard identity/coverage mismatch')
        selected=set(selected)
        seen=set()
        for row in part['partitions']:
            identity=(row['path'],row['bucket'])
            obj=dataset.objects.get(row['path'],{})
            if (row['path'] not in selected or identity in seen
                    or row['bucket'] not in {b[0] for b in BUCKETS}
                    or (row['dataset'],row['category'])!=(obj.get('dataset'),obj.get('category'))
                    or not 0<=row['lo']<row['hi']<=obj['document_count']):
                raise ValueError('invalid catalog shard row')
            seen.add(identity)
        rows.extend(part['partitions'])
    order={b[0]:i for i,b in enumerate(BUCKETS)}
    rows.sort(key=lambda row:(row['path'],order[row['bucket']]))
    return {'format':CATALOG_FORMAT,'dataset':dataset.pin,'buckets':[list(x) for x in BUCKETS],'partitions':rows}
