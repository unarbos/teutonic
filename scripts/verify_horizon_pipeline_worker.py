"""GPU acceptance of the production Horizon RPC against saved deterministic losses.

Reads local model/fixtures only. Runs one fresh worker process and leaves all
production directories and remote object storage untouched.
"""

import argparse
import json
import math
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from teutonic.evaluator.horizon_pipeline_runtime import HorizonWorker, runtime_identity
from teutonic.evaluator.vllm_runtime import wait_for_gpu_release


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--small-batches-only', action='store_true',
                   help='Check short and mixed-length groups of 1/2/3/8; skip the throughput cases')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads(args.fixture.read_text())
    baseline = np.load(args.baseline)
    expected = [math.fsum(baseline[f'doc-{i:03d}-repeat-0'].tolist()) for i in range(128)]
    rows = [{'id': f'doc-{i:03d}', 'tokens': fixture[i % len(fixture)]['tokens']}
            for i in range(128)]
    runtime = runtime_identity()
    report = {'runtime': runtime, 'cases': [], 'workers': [], 'passed': False,
              'scope': 'small-batch-repeatability' if args.small_batches_only else 'full-acceptance'}
    stop = threading.Event()
    peaks = [0] * 8

    def save():
        report['peak_used_mib'] = list(peaks)
        (args.output/'report.json').write_text(json.dumps(report, indent=2))

    def monitor():
        with (args.output/'gpu.jsonl').open('w') as log:
            while not stop.is_set():
                try:
                    sample = subprocess.run(
                        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
                    values = [[int(v.strip()) for v in line.split(',')] for line in sample.stdout.splitlines()]
                    for index, used, _ in values:
                        peaks[index] = max(peaks[index], used)
                    log.write(json.dumps({'time': time.time(), 'gpus': values})+'\n')
                    log.flush()
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    pass
                stop.wait(2)

    telemetry = threading.Thread(target=monitor, daemon=True)
    telemetry.start()
    try:
        for process in range(1):
            wait_for_gpu_release(list(range(8)), timeout=180)
            with HorizonWorker('horizon', args.snapshot, list(range(8)), args.output, runtime,
                               lambda event: print(json.dumps(event), flush=True),
                               worker_id=f'horizon-{process}') as worker:
                worker.wait_ready()

                def run(name, groups, process=process, worker=worker):
                    started = time.monotonic()
                    before = worker.report['scoring_seconds']
                    values = []
                    for group in groups:
                        docs = [SimpleNamespace(doc_id=r['id']) for r in group]
                        values.extend(worker.score(docs, [r['tokens'] for r in group]))
                    elapsed = time.monotonic()-started
                    deltas = [r['nll']-expected[int(r['id'].split('-')[1]) % 128] for r in values]
                    count = sum(r['scored_tokens'] for r in values)
                    scoring = worker.report['scoring_seconds']-before
                    case = {'process': process, 'name': name, 'documents': len(values),
                            'scored_tokens': count, 'wall_seconds': elapsed, 'scoring_seconds': scoring,
                            'scored_tokens_per_second': count/scoring,
                            'wall_tokens_per_second': count/elapsed,
                            'different_document_nll': sum(d != 0 for d in deltas),
                            'max_abs_document_nll_delta': max(map(abs, deltas), default=0)}
                    report['cases'].append(case)
                    (args.output/f'{process}-{name}.json').write_text(json.dumps(values))
                    save()
                    print(json.dumps(case), flush=True)
                    if any(deltas):
                        raise AssertionError(f'Production worker differs from deterministic baseline: {case}')

                run('initial', [rows])
                run('repeat', [rows])
                run('reverse', [list(reversed(rows))])
                # Different request boundaries change padding, GEMM shapes and grouping.
                if not args.small_batches_only:
                    run('groups-16', [rows[i:i+16] for i in range(0, 128, 16)])
                # Include small, mixed-length requests that exercise minimum padding.
                ordered = sorted(rows[:min(len(fixture), 128)], key=lambda row: len(row['tokens']))
                short = ordered[:8]
                mixed = [ordered[i * (len(ordered)-1) // 7] for i in range(8)]
                report['small_batch_document_lengths'] = {
                    label: {r['id']: len(r['tokens']) for r in selected}
                    for label, selected in (('short', short), ('mixed', mixed))}
                for label, selected in (('short', short), ('mixed', mixed)):
                    for size in (1, 2, 3, 8):
                        groups = [selected[i:i+size] for i in range(0, len(selected), size)]
                        run(f'{label}-groups-{size}', groups)
                        run(f'{label}-groups-{size}-repeat', groups)
                if process == 0 and not args.small_batches_only:
                    run('individual', [[rows[i]] for i in [0, 1, 7, 31, 63, 90, 127]])
                    large = [{'id': f'doc-{i:04d}', 'tokens': rows[i % 128]['tokens']} for i in range(1024)]
                    run('large-1024', [large])
                    run('large-1024-repeat', [large])
                report['workers'].append(dict(worker.report))
                save()
            wait_for_gpu_release(list(range(8)), timeout=180)
        report['passed'] = True
        save()
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        save()
        raise
    finally:
        stop.set()
        telemetry.join(timeout=20)


if __name__ == '__main__':
    main()
