#!/usr/bin/env python3
"""Offline Horizon batching experiment; never changes production settings or R2.

Run each configuration in a fresh process. The 128-document fixture repeats
some of the saved real documents to match the production RPC batch size.
Compare every scored token, not just aggregate loss. Memory sampling is owned
by the external controller so it includes engine subprocess allocations.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path


class AllocatorTelemetry:
    def benchmark_allocator_stats(self):
        import torch
        stats = torch.cuda.memory_stats()
        return {key: stats.get(key, 0) for key in (
            'allocated_bytes.all.peak', 'reserved_bytes.all.peak',
            'allocated_bytes.all.current', 'reserved_bytes.all.current',
            'num_alloc_retries', 'num_ooms')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-tokens', type=int, required=True)
    parser.add_argument('--sequences', type=int, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--stress', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    from teutonic.evaluator.vllm_score import deterministic_env, identity, settings

    os.environ.update(deterministic_env('horizon'))
    import numpy as np
    from vllm import LLM, SamplingParams

    source = json.loads(args.inputs.read_text())
    if not source or any(r['tokens'][-1] != 1 for r in source):
        raise ValueError('Native Horizon EOS required')
    rows = [{**source[i % len(source)], 'id': f'doc-{i:03d}'} for i in range(128)]
    configuration = settings('horizon')
    configuration.update(max_num_batched_tokens=args.batch_tokens, max_num_seqs=args.sequences)
    configuration['worker_extension_cls'] = 'benchmark_horizon_batching.AllocatorTelemetry'
    metadata = {'settings': configuration, 'runtime': identity(),
                'allocator_config': os.environ.get('PYTORCH_ALLOC_CONF', ''),
                'input_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                'source_documents': len(source), 'request_documents': len(rows),
                'fixture': '128 entries cycling saved real documents; prefix caching disabled'}
    (args.output / 'configuration.json').write_text(json.dumps(metadata, indent=2))

    def emit(**values):
        print(json.dumps(values), flush=True)

    emit(phase='loading', settings=configuration)
    started = time.monotonic()
    llm = LLM(model=str(args.snapshot), **configuration)
    load_seconds = time.monotonic() - started
    emit(phase='loaded', seconds=load_seconds)
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0,
                            detokenize=False, seed=42)
    baseline = np.load(args.baseline) if args.baseline else None
    reference = None
    cases = []

    def score(name, groups, compare=True):
        started = time.monotonic()
        values = {}
        for group in groups:
            outputs = llm.generate([{'prompt_token_ids': r['tokens']} for r in group],
                                   params, use_tqdm=False)
            for row, output in zip(group, outputs, strict=True):
                lp = output.prompt_logprobs
                if (list(output.prompt_token_ids) != row['tokens'] or lp is None
                        or len(lp) != len(row['tokens']) or lp[0]):
                    raise AssertionError('Input or target accounting changed')
                losses = np.asarray([-p[t].logprob for p, t in
                                     zip(lp[1:], row['tokens'][1:], strict=True)], dtype=np.float64)
                if not np.all(np.isfinite(losses)) or np.any(losses < 0):
                    raise AssertionError('Invalid NLL')
                values[row['id']] = losses
        elapsed = time.monotonic() - started
        compared = reference if reference is not None else baseline
        diffs = [v - compared[k] for k, v in values.items()] if compare and compared is not None else []
        case = {'name': name, 'seconds': elapsed, 'scored_tokens': sum(len(v) for v in values.values()),
                'different_tokens': sum(int(np.count_nonzero(d)) for d in diffs),
                'max_token_nll_difference': max((float(np.max(np.abs(d))) for d in diffs), default=0),
                'nll': math.fsum(math.fsum(v) for v in values.values())}
        case['tokens_per_second'] = case['scored_tokens'] / elapsed
        case['allocator_stats'] = llm.collective_rpc('benchmark_allocator_stats', timeout=60)
        cases.append(case)
        emit(phase='scored', **case)
        (args.output / 'cases.json').write_text(json.dumps(cases, indent=2))
        return values

    reference = score('warmup', [rows])
    np.savez(args.output / 'token-nll.npz', **reference)
    score('repeat-1', [rows])
    score('repeat-2', [rows])
    score('reverse', [list(reversed(rows))])
    if args.stress:
        ordered = sorted(rows, key=lambda r: len(r['tokens']))
        score('length-sorted', [ordered])
        score('individual', [[ordered[i]] for i in [0, len(ordered)//2, len(ordered)-1]])
        longest = max(rows, key=lambda r: len(r['tokens']))
        stress_rows = [{**longest, 'id': f'stress-{i:03d}'} for i in range(128)]
        stress = score('128-longest-documents', [stress_rows], compare=False)
        differences = [v - reference[longest['id']] for v in stress.values()]
        cases[-1]['different_tokens'] = sum(int(np.count_nonzero(d)) for d in differences)
        cases[-1]['max_token_nll_difference'] = max(float(np.max(np.abs(d))) for d in differences)
        emit(phase='stress_comparison', **cases[-1])
    warm = [c['tokens_per_second'] for c in cases if c['name'] in ('repeat-1', 'repeat-2', 'reverse')]
    report = {**metadata, 'load_seconds': load_seconds, 'cases': cases,
              'median_tokens_per_second': statistics.median(warm),
              'exact': all(c['different_tokens'] == 0 for c in cases)}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    emit(phase='complete', median_tokens_per_second=report['median_tokens_per_second'], exact=report['exact'])
    if not report['exact']:
        raise AssertionError('Token losses changed within or across configurations')


if __name__ == '__main__':
    main()
