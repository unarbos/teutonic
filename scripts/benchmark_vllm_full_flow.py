"""Read-only CodeIO UTF-8 -> Horizon -> paired MiMo -> reference-reward benchmark."""
import argparse
import hashlib
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path('/home/ubuntu/teutonic-test-update')
HORIZON = ROOT/'cache/horizon/7d22cc0ac26a91003d19482efa71070244239b11'
MODELS = {
    'king': ROOT/'cache/mix07-vllm/bf49ff86b87a2b14126a0fe9a021917bd1451968ffb378c5659ac93bdaeb7eb6',
    'challenger': ROOT/'cache/mix07-vllm/de0c6076d5cc463765708e4961ea571e684eb6055c914b017eecf4194b343e06',
}


def main():
    from scripts.verify_r2_category_duel_gpu import GetOnlyObjects
    from teutonic.evaluation.packed_text import (
        DatasetReader,
        PackedDataset,
        canonical,
        digest,
        validate_mix,
    )
    from teutonic.evaluation.reference_policy import reference_verdict
    from teutonic.evaluator.document_duel import TEXT_POLICY, ReferenceCache
    from teutonic.evaluator.horizon import CONFIG_SHA256, INDEX_SHA256, REVISION, file_hash
    from teutonic.evaluator.text_pipeline import TextTokenizer, prepared_batches
    from teutonic.evaluator.text_sampling import sample_documents

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--token-budget', type=int, default=35000000)
    parser.add_argument('--inventory-dir', type=Path, default=ROOT/'results/codeio-70m-live-r2-20261111T205700Z/inventory')
    args = parser.parse_args()
    run = args.output_dir.resolve()
    if not run.is_relative_to(ROOT) or args.token_budget <= 0:
        raise ValueError('Test output must stay inside the isolated checkout; budget must be positive')
    run.mkdir(parents=True, exist_ok=False)
    for line in Path('/home/ubuntu/teutonic/.gpu.env').read_text().splitlines():
        key, separator, value = line.removeprefix('export ').partition('=')
        if separator and key.strip().startswith('TEUTONIC_EVALUATOR_R2_'):
            parts = shlex.split(value, comments=True)
            os.environ[key.strip()] = parts[0] if parts else ''
    def save(name, value):
        (run/name).write_text(json.dumps(value, indent=2)+'\n')
    started = time.monotonic()
    def emit(**values):
        event = {'elapsed_seconds': time.monotonic()-started,
                 'utc': datetime.now(UTC).isoformat(), **values}
        with (run/'phases.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    def capture(name, command):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        (run/name).write_text(result.stdout+result.stderr)
        return result
    def idle():
        deadline = time.monotonic()+120
        while True:
            result = capture('gpu-current.csv', ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'])
            if result.returncode:
                raise RuntimeError('GPU health query failed')
            if all(int(line.split(',')[1]) < 1000 for line in result.stdout.splitlines()):
                return
            if time.monotonic() > deadline:
                raise RuntimeError('GPUs did not become idle; no unrelated process was stopped')
            time.sleep(2)

    idle()
    shutil.copy2(run/'gpu-current.csv', run/'gpu-before.csv')
    capture('kernel-before.log', ['journalctl', '-k', '-b', '--no-pager', '--grep', 'NVRM: Xid|SXid'])
    for name, expected in [('config.json', CONFIG_SHA256), ('model.safetensors.index.json', INDEX_SHA256)]:
        if file_hash(HORIZON/name) != expected:
            raise ValueError('Horizon checkpoint pin differs')
    if file_hash(MODELS['king']/'tokenizer.json') != file_hash(MODELS['challenger']/'tokenizer.json'):
        raise ValueError('King/challenger tokenizers differ')
    inventory = run/'inventory'
    shutil.copytree(args.inventory_dir, inventory)
    mix = json.loads((inventory/'mix.json').read_text())
    mix['token_budget'] = args.token_budget
    mix['competition'] = 'codeio-full-vllm-readonly-test'
    validate_mix(mix)
    (inventory/'mix.json').write_bytes(canonical(mix))
    seed_material = 'block_hash=0x'+'a'*64+'|hotkey=codeio-full-vllm-35m'
    seed = int.from_bytes(hashlib.blake2b(seed_material.encode(), digest_size=8).digest(), 'little')
    save('run.json', {'started_utc': datetime.now(UTC).isoformat(), 'token_budget': args.token_budget,
                      'seed_material': seed_material, 'dataset_seed': seed, 'bootstrap_seed': 0,
                      'models': {r: str(p) for r, p in MODELS.items()}, 'horizon': str(HORIZON),
                      'reference_cache': 'new and empty; no previous reference losses reused',
                      'dataset': 'CodeIO only, packed UTF-8 live R2 ranges',
                      'inventory': 'reuse pinned, previously verified category inventory; validate ETags/hashes on reads',
                      'remote_writes': 0, 'driver': Path('/proc/driver/nvidia/version').read_text()})
    class RetriedObjects(GetOnlyObjects):
        def __init__(self, directory):
            super().__init__(directory)
            self.object_locks = {}
            self.retry_count = 0

        def get_object(self, **kwargs):
            from botocore.exceptions import ClientError

            key = kwargs['Key']
            if key in self.local:
                return super().get_object(**kwargs)
            with self.lock:
                lock = self.object_locks.setdefault(key, threading.Lock())
            # Hold the per-object lock until the complete bounded range is read.
            with lock:
                for attempt in range(8):
                    try:
                        response = super().get_object(**kwargs)
                        with response['Body'] as body:
                            size = response['ContentLength']
                            if not kwargs.get('Range') or size > 64*1024**2:
                                raise ValueError('Unbounded remote read in benchmark')
                            raw = body.read(size+1)
                        response['Body'] = io.BytesIO(raw)
                        return response
                    except ClientError as error:
                        status = error.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
                        code = error.response.get('Error', {}).get('Code')
                        if attempt == 7 or (status not in {429, 500, 502, 503, 504}
                                            and code not in {'ServiceUnavailable', 'SlowDown'}):
                            raise
                        delay = min(30, 2**attempt)
                        with self.lock:
                            self.retry_count += 1
                        emit(phase='r2_retry', key=key, status=status, code=code,
                             attempt=attempt+1, delay_seconds=delay)
                        time.sleep(delay)
                raise AssertionError('unreachable')

    client = RetriedObjects(inventory)
    processes = []
    timings, reports, expected = {}, {}, {}
    env_base = {**os.environ, 'PATH': str(ROOT/'.venv-vllm/bin')+os.pathsep+os.environ['PATH'],
                'VLLM_BATCH_INVARIANT': '1', 'VLLM_USE_V2_MODEL_RUNNER': '0',
                'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_NO_USAGE_STATS': '1',
                'PYTHONHASHSEED': '0', 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '8'}
    env_base = {k: v for k, v in env_base.items() if 'R2_' not in k and k != 'TEUTONIC_MIMO_TRACE_DIR'}

    def launch(role):
        horizon = role == 'horizon'
        env = {**env_base, 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7' if horizon else ('0,1,2,3' if role == 'king' else '4,5,6,7'),
               'VLLM_PLUGINS': '' if horizon else 'teutonic_mimo',
               'VLLM_USE_FUSED_MOE_GROUPED_TOPK': '1' if horizon else '0',
               'FLASHINFER_WORKSPACE_BASE': str(ROOT/('results/horizon-vllm-determinism/native-v1-runner/flashinfer-workspace'
                                                    if horizon else 'results/mimo-vllm-validation/flashinfer-workspace'))}
        command = [str(ROOT/'.venv-vllm/bin/python'), '-u', str(ROOT/'scripts/benchmark_vllm_text_worker.py'),
                   '--role', role, '--snapshot', str(HORIZON if horizon else MODELS[role]), '--output-dir', str(run/role)]
        if horizon:
            command += ['--validation-inputs', str(ROOT/'results/horizon-vllm-determinism/inputs.json'),
                        '--validation-losses', str(ROOT/'results/horizon-vllm-determinism/native-v1-runner/first/batched-token-nll.json')]
        log = (run/f'{role}.log').open('w')
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdin=subprocess.PIPE, text=True,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((role, process, log))
        emit(phase='worker_started', role=role, pid=process.pid)
        return process

    def stream_inputs(dataset, documents, tokenizer, roles, maximum):
        jobs = [launch(role) for role in roles]
        counts = {}
        batches = prepared_batches(dataset, documents, tokenizer, batch_documents=128,
                                   prefetch_batches=2, max_tokens=maximum)
        for group, tokens in batches:
            # Keep both engines on exactly the same bounded batch of native IDs.
            rows = [{'id': doc.doc_id, 'tokens': ids} for doc, ids in zip(group, tokens, strict=True)]
            line = json.dumps(rows)+'\n'
            for job in jobs:
                job.stdin.write(line)
                job.stdin.flush()
            counts.update((row['id'], len(row['tokens'])-1) for row in rows)
        for job in jobs:
            job.stdin.close()
        for role, job in zip(roles, jobs, strict=True):
            code = job.wait()
            emit(phase='worker_exited', role=role, returncode=code)
            if code:
                raise RuntimeError(f'{role} worker failed with exit code {code}')
            values = [json.loads(line) for line in (run/role/'scores.jsonl').read_text().splitlines()]
            if [v['id'] for v in values] != [d.doc_id for d in documents]:
                raise ValueError('Scored documents differ from selection or ordering')
            if any(v['scored_tokens'] != counts[v['id']] for v in values):
                raise ValueError('Scored-token count differs from prepared inputs')
            expected[role] = values
            reports[role] = json.loads((run/role/'report.json').read_text())
        if len(roles) == 2 and any(k['input_sha256'] != c['input_sha256'] for k, c in zip(
                expected['king'], expected['challenger'], strict=True)):
            raise ValueError('King/challenger inputs differ')

    try:
        with tempfile.TemporaryDirectory(prefix='text-', dir=run) as scratch:
            reader = DatasetReader(client)
            dataset = PackedDataset(reader, mix['dataset'], text_cache=scratch)
            catalog = reader.json(mix['sampling'])
            sample_start = time.monotonic()
            emit(phase='sampling')
            documents, metadata = sample_documents(dataset, catalog, mix, seed)
            timings['sampling_seconds'] = time.monotonic()-sample_start
            save('selection.json', {'metadata': metadata, 'documents': [asdict(d) for d in documents]})
            emit(phase='documents_selected', documents=len(documents), original_mimo_tokens=sum(d.original_mimo_token_count for d in documents),
                 digest=metadata['digest'], seconds=timings['sampling_seconds'])
            phase_start = time.monotonic()
            htok = TextTokenizer(HORIZON, backend='gigatoken')
            emit(phase='reference_start', misses=len(documents))
            stream_inputs(dataset, documents, htok, ['horizon'], 524287)
            timings['reference_phase_seconds'] = time.monotonic()-phase_start
            if (run/'reference-losses.sqlite').exists():
                raise AssertionError('Expected a new reference-loss cache file')
            cache = ReferenceCache(run/'reference-losses.sqlite')
            identity = json.loads((run/'horizon/configuration.json').read_text())
            try:
                for doc, result in zip(documents, expected['horizon'], strict=True):
                    key = digest({'checkpoint': REVISION, 'text': doc.text_sha256, 'bytes': doc.byte_length,
                                  'text_policy': TEXT_POLICY, 'backend': 'vllm-native-bf16-tp8-ep-test', 'identity': identity})
                    cached = cache.get(key)
                    if cached is None:
                        cache.put(key, result['nll'], result['scored_tokens'])
                    elif cached != (result['nll'], result['scored_tokens']):
                        raise AssertionError('Identical text produced different reference scores')
            finally:
                cache.close()
            switch_start = time.monotonic()
            idle()
            timings['switch_seconds'] = time.monotonic()-switch_start
            phase_start = time.monotonic()
            mtok = TextTokenizer(MODELS['king'], backend='gigatoken', eos_token_id=151645)
            emit(phase='paired_start')
            stream_inputs(dataset, documents, mtok, ['king', 'challenger'], 16384)
            timings['paired_phase_seconds'] = time.monotonic()-phase_start
            bootstrap_start = time.monotonic()
            verdict = reference_verdict(*[[v['nll'] for v in expected[role]] for role in ('king', 'challenger', 'horizon')],
                                        [d.byte_length for d in documents], bootstrap_seed=0,
                                        n_bootstrap=10000, alpha=.001, delta_threshold=.0015)
            timings['bootstrap_seconds'] = time.monotonic()-bootstrap_start
            report = {'completed': True, 'dataset': metadata, 'documents': len(documents),
                      'original_mimo_tokens': sum(d.original_mimo_token_count for d in documents),
                      'bytes': sum(d.byte_length for d in documents), 'timings': timings, 'workers': reports,
                      'verdict': verdict, 'r2_reads': {**client.stats, 'retries': client.retry_count}, 'reference_misses': len(documents),
                      'tokenizers': {'horizon': htok.status(), 'mimo': mtok.status()},
                      'scope': 'CodeIO-only full UTF-8 flow; real MIX07 checkpoint pair; timing test, no production promotion'}
        idle()
        timings['total_wall_seconds'] = time.monotonic()-started
        report['temporary_text_cache_cleaned'] = not any(run.glob('text-*'))
        save('report.json', report)
        save('passed.json', {'passed': True, 'seconds': timings['total_wall_seconds']})
        emit(phase='complete', timings=timings, verdict=verdict)
    except BaseException as error:
        save('failure.json', {'type': type(error).__name__, 'error': str(error), 'elapsed_seconds': time.monotonic()-started})
        raise
    finally:
        for _role, process, log in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()
        client.client.close()
        capture('kernel-after.log', ['journalctl', '-k', '-b', '--no-pager', '--grep', 'NVRM: Xid|SXid'])
        capture('gpu-after.csv', ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'])


if __name__ == '__main__':
    main()
