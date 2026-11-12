#!/usr/bin/env python3
"""Instrument the supplied Horizon pipeline for token-level repeatability tests.

No model math is replaced. Capture CE values before document reduction, reuse
the loaded model for request-pattern checks, and optionally compare to a prior
fresh process. Torch deterministic-algorithm mode is tested separately.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--vllm-baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(False)
    original = args.source.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != '94fc9c65be8b26f12cf8d36830a4b26c501e4058026ed19ed1ad1952cd72b551':
        raise ValueError('Supplied pipeline source changed; inspect before instrumenting')
    marker = 'threads = [threading.Thread(target=stage'
    if original.count(marker) != 1:
        raise ValueError('Unexpected supplied source layout')
    source = original.split(marker)[0]
    replacements = {
        '            nll = torch.zeros(Bn, dtype=torch.float64, device=dev)':
            '            token_nll = torch.empty_like(tv, dtype=torch.float32)\n'
            '            nll = torch.zeros(Bn, dtype=torch.float64, device=dev)',
        '                nll.index_add_(0, doc[s:s + args.head_chunk], ce.double())':
            '                token_nll[s:s + args.head_chunk] = ce\n'
            '                nll.index_add_(0, doc[s:s + args.head_chunk], ce.double())',
        '            results.put((bid, nll.cpu().numpy()))':
            '            results.put((bid, nll.cpu().numpy(), token_nll.cpu().numpy(), lens))',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError('Unexpected scoring source layout')
        source = source.replace(before, after)
    (args.output/'instrumented-source.py').write_text(source)
    metadata = {'original_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                'instrumented_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'versions': {k:version(k) for k in ['torch','transformers','accelerate']},
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'allocator': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                'seed':42, 'documents':128, 'cases':[]}
    (args.output/'configuration.json').write_text(json.dumps(metadata, indent=2))
    sys.argv = [str(args.source),'--model',str(args.snapshot),'--eval-dir',str(args.inputs),
                '--out',str(args.output/'unused-original-output'),'--batch-tokens','131072']
    ns = {'__name__':'horizon_pipeline_instrumented','__file__':str(args.source)}
    started = time.monotonic()
    exec(compile(source, str(args.source), 'exec'), ns)  # noqa: S102 - pinned local benchmark source
    metadata['load_and_repack_seconds'] = time.monotonic()-started
    fixture = json.loads(args.fixture.read_text())
    rows = ns['rows'][:128]
    lookup = {r['sample_id']:r for r in rows}
    for i, row in enumerate(rows):
        expected = fixture[i % len(fixture)]['tokens']
        actual = ns['tokens'][row['tok_offset']:row['tok_offset']+row['length']]
        if actual.tolist() != expected or expected[-1] != 1:
            raise AssertionError('Fixture differs from vLLM inputs/native EOS')
    baseline_file = np.load(args.vllm_baseline)
    baseline = {r['sample_id']: baseline_file[f'doc-{i:03d}'] for i,r in enumerate(rows)}
    byte_lengths = {r['sample_id']:fixture[i%len(fixture)]['byte_length'] for i,r in enumerate(rows)}
    report = {**metadata, 'cases':[]}

    def differences(values, expected):
        deltas = {k:v.astype(np.float64)-expected[k] for k,v in values.items()}
        nll = {k:math.fsum(d) for k,d in deltas.items()}
        return {'different_tokens':sum(int(np.count_nonzero(d)) for d in deltas.values()),
                'max_abs_token_nll_delta':max(float(np.max(np.abs(d))) for d in deltas.values()),
                'aggregate_nats_per_token_delta':math.fsum(nll.values())/sum(len(v) for v in values.values()),
                'aggregate_bits_per_byte_delta':math.fsum(nll.values())/(math.log(2)*sum(byte_lengths[k] for k in values)),
                'max_abs_document_nats_per_token_delta':max(abs(nll[k])/len(values[k]) for k in values),
                'max_abs_document_bits_per_byte_delta':max(abs(nll[k])/(math.log(2)*byte_lengths[k]) for k in values)}

    def run(name, requested, token_budget=131072, max_batch=128, sort=True, reference=None, previous=None):
        ordered = sorted(requested,key=lambda r:r['length']) if sort else list(requested)
        batches, current = [], []
        for row in ordered:
            longest = max([r['length'] for r in current]+[row['length']])
            if current and (len(current)>=max_batch or (len(current)+1)*longest>token_budget):
                batches.append(current)
                current=[]
            current.append(row)
        if current:
            batches.append(current)
        if sort:
            batches.reverse()
        ns.update(batches=batches, item_ids={}, errors=[],
                  queues=[ns['queue'].Queue(maxsize=ns['args'].queue_depth) for _ in range(ns['NG'])],
                  results=ns['queue'].Queue())
        workers=[threading.Thread(target=ns['stage'],args=(g,),daemon=True) for g in range(ns['NG'])]
        for worker in workers:
            worker.start()
        started=time.monotonic()
        feeder=threading.Thread(target=ns['feeder'],daemon=True)
        feeder.start()
        values, sums = {}, {}
        while True:
            result=ns['results'].get(timeout=900)
            if result is None:
                break
            if result[0]=='ERROR':
                raise RuntimeError(result[1])
            bid, reduced, losses, lens=result
            offset=0
            for row, length, total in zip(batches[bid],lens,reduced,strict=True):
                values[row['sample_id']]=losses[offset:offset+length-1].copy()
                sums[row['sample_id']]=float(total)
                offset+=length-1
            if offset!=len(losses):
                raise AssertionError('Token split accounting changed')
            del ns['item_ids'][bid]
        for worker in [*workers,feeder]:
            worker.join(timeout=30)
            if worker.is_alive():
                raise RuntimeError('Pipeline worker failed to exit')
        elapsed=time.monotonic()-started
        if set(values)!={r['sample_id'] for r in requested}:
            raise AssertionError('Missing or duplicate output')
        for key,value in values.items():
            if len(value)!=lookup[key]['length']-1 or not np.all(np.isfinite(value)) or np.any(value<0):
                raise AssertionError('Invalid target losses')
        np.savez(args.output/f'{name}.npz',**values)
        (args.output/f'{name}-document-nll.json').write_text(json.dumps(sums))
        case={'name':name, 'seconds':elapsed,'scored_tokens':sum(len(v) for v in values.values()),
              'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
              'microbatches':len(batches),'max_padded_tokens':token_budget,'max_batch':max_batch,
              'sorted':sort,'vs_vllm':differences(values,baseline),
              'max_reduction_vs_fsum':max(abs(sums[k]-math.fsum(v)) for k,v in values.items())}
        case['tokens_per_second']=case['scored_tokens']/elapsed
        if reference is not None:
            case['vs_same_process_initial']=differences(values,reference)
        if previous is not None:
            case['vs_fresh_process']=differences(values,np.load(previous))
        report['cases'].append(case)
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'phase':'case_complete',**case}),flush=True)
        return values

    reference=run('initial',rows,previous=args.previous/'initial.npz' if args.previous else None)
    run('repeat',rows,reference=reference)
    if not args.previous:
        run('reverse-unsorted',list(reversed(rows)),sort=False,reference=reference)
        run('smaller-batches',rows,token_budget=65536,max_batch=16,reference=reference)
        ordered=sorted(rows,key=lambda r:r['length'])
        run('individual',[ordered[i] for i in [0,len(rows)//2,len(rows)-1]],max_batch=1,reference=reference)
    torch.use_deterministic_algorithms(True)
    deterministic=run('deterministic-initial',rows,
                      previous=args.previous/'deterministic-initial.npz' if args.previous else None)
    run('deterministic-repeat',rows,reference=deterministic)
    run('deterministic-smaller-batches',rows,token_budget=65536,max_batch=16,reference=deterministic)
    report['completed']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'phase':'complete','cases':len(report['cases'])}),flush=True)


if __name__=='__main__':
    main()
