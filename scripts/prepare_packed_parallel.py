"""CPU-only forked catalog preparation with per-process resumable checkpoints."""

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
import tomllib
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.prepare_packed_evaluation import prepare
from teutonic.competitions import ACTIVE_COMPETITIONS
from teutonic.evaluation.packed_catalog import build_resumable_catalog, merge_catalogs
from teutonic.evaluation.packed_text import DatasetReader, PackedDataset, canonical, read_client


def worker(dataset,output,index,count,threads,page_bytes):
    dataset.reader=DatasetReader(read_client(max_pool_connections=threads))
    def progress(value):
        os.write(1,(json.dumps({'shard':index,**value})+'\n').encode())
    catalog=build_resumable_catalog(dataset,output/f'catalog-{index:02d}-of-{count}.sqlite',
                                    workers=threads,page_bytes=page_bytes,on_progress=progress,
                                    shard_index=index,shard_count=count)
    target=output/f'catalog-{index:02d}-of-{count}.json'
    temporary=target.with_suffix('.pending')
    temporary.write_bytes(canonical(catalog))
    temporary.replace(target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-pin',type=Path,required=True)
    p.add_argument('--dataset-manifest-file',type=Path,required=True)
    p.add_argument('--definitions',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--processes',type=int,default=16)
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--page-bytes',type=int,default=65536)
    args=p.parse_args()
    if not 2<=args.processes<=64:
        p.error('processes must be in 2..64')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pin=json.loads(args.dataset_pin.read_text())
    class LocalReader:
        def json(self,requested):
            raw=args.dataset_manifest_file.read_bytes()
            if requested!=pin or hashlib.sha256(raw).hexdigest()!=pin['sha256']:
                raise ValueError('dataset manifest hash mismatch')
            return json.loads(raw)
    dataset=PackedDataset(LocalReader(),pin)
    config=tomllib.loads(args.definitions.read_text())
    definitions={key:config['competitions'][key] for key in ACTIVE_COMPETITIONS}
    if any(d['long_document_tokens']!=35000000 for d in definitions.values()):
        raise ValueError('expected the approved 35M rollout definitions')
    context=multiprocessing.get_context('fork')
    processes=[]
    try:
        for index in range(args.processes):
            process=context.Process(target=worker,args=(dataset,args.output_dir,index,args.processes,
                                                        args.workers,args.page_bytes))
            process.start()
            processes.append(process)
        while any(process.is_alive() for process in processes):
            if any(process.exitcode not in (None,0) for process in processes):
                raise RuntimeError('catalog worker failed; checkpoints retained for resume')
            time.sleep(1)
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError('catalog worker failed')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
    parts=[json.loads((args.output_dir/f'catalog-{i:02d}-of-{args.processes}.json').read_bytes())
           for i in range(args.processes)]
    catalog=merge_catalogs(dataset,parts)
    entries=prepare(dataset,definitions,config['categories'],args.output_dir,catalog=catalog)
    print(json.dumps({'state':'prepared','files':len(entries),'partitions':len(catalog['partitions']),
                      'remote_writes':0}),flush=True)


if __name__=='__main__':
    main()
