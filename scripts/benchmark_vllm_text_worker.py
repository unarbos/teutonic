"""Isolated streaming vLLM worker for the read-only full-flow GPU benchmark."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['horizon', 'king', 'challenger'], required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validation-inputs', type=Path)
    parser.add_argument('--validation-losses', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    import numpy as np
    from vllm import LLM, SamplingParams

    horizon = args.role == 'horizon'
    eos = 1 if horizon else 151645
    settings = {
        'model': str(args.snapshot), 'trust_remote_code': True, 'model_impl': 'vllm',
        'dtype': 'bfloat16', 'tensor_parallel_size': 8 if horizon else 4,
        'enable_expert_parallel': True, 'max_model_len': 524288 if horizon else 16385,
        'max_num_seqs': 16, 'max_num_batched_tokens': 16384 if horizon else 8192,
        'kv_cache_memory_bytes': (32 if horizon else 16) * 1024**3,
        'gpu_memory_utilization': 0.8, 'seed': 42, 'skip_tokenizer_init': True,
        'enable_prefix_caching': False, 'enforce_eager': True,
        'disable_custom_all_reduce': False,
    }
    if not horizon:
        settings['attention_config'] = {'backend': 'TRITON_ATTN_DIFFKV'}
    configuration = {'role': args.role, 'settings': settings,
                     'environment': {k: os.environ.get(k) for k in (
                         'VLLM_BATCH_INVARIANT', 'VLLM_USE_FUSED_MOE_GROUPED_TOPK',
                         'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_PLUGINS')},
                     'versions': {p: version(p) for p in ('vllm', 'torch', 'transformers')},
                     'scorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'snapshot_hashes': {name: hashlib.sha256((args.snapshot/name).read_bytes()).hexdigest()
                                         for name in ('config.json', 'tokenizer.json', 'model.safetensors.index.json')}}
    (args.output_dir/'configuration.json').write_text(json.dumps(configuration, indent=2)+'\n')
    def emit(**values):
        event = {'role': args.role, 'monotonic': time.monotonic(), **values}
        with (args.output_dir/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    started = time.monotonic()
    emit(phase='loading')
    llm = LLM(**settings)
    load_seconds = time.monotonic()-started
    emit(phase='loaded', seconds=load_seconds)
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0, detokenize=False, seed=42)

    def score(rows):
        if not rows or any(not 2 <= len(r['tokens']) < settings['max_model_len'] or r['tokens'][-1] != eos for r in rows):
            raise ValueError('Invalid document context length or EOS; no truncation is allowed')
        outputs = llm.generate([{'prompt_token_ids': r['tokens']} for r in rows], params, use_tqdm=False)
        values = []
        for row, output in zip(rows, outputs, strict=True):
            tokens, logprobs = row['tokens'], output.prompt_logprobs
            if list(output.prompt_token_ids) != tokens or len(logprobs) != len(tokens) or logprobs[0]:
                raise ValueError('Prompt/target accounting changed')
            losses = np.asarray([-entry[token].logprob for token, entry in zip(tokens[1:], logprobs[1:], strict=True)], dtype='<f8')
            if not np.isfinite(losses).all() or (losses < 0).any():
                raise ValueError('Invalid token NLL')
            values.append({'id': row['id'], 'nll': math.fsum(losses), 'scored_tokens': len(losses),
                           'input_sha256': hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest(),
                           'token_loss_sha256': hashlib.sha256(losses.tobytes()).hexdigest()})
        return values

    validation_seconds = 0.0
    if args.validation_inputs:
        validation_start = time.monotonic()
        fixtures = json.loads(args.validation_inputs.read_text())
        expected = json.loads(args.validation_losses.read_text())
        observed = score(fixtures)
        changed = [v['id'] for v in observed if v['token_loss_sha256'] != hashlib.sha256(
            np.asarray(expected[v['id']], dtype='<f8').tobytes()).hexdigest()]
        validation_seconds = time.monotonic()-validation_start
        validation = {'documents': len(observed), 'scored_tokens': sum(v['scored_tokens'] for v in observed),
                      'different_documents': changed, 'seconds': validation_seconds,
                      'purpose': 'Verify the smaller KV allocation and full context limit against previously validated vLLM token losses'}
        (args.output_dir/'configuration-validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        emit(phase='configuration_validated', **validation)
        if changed:
            raise AssertionError('Changed Horizon allocation/context configuration changes validated token losses')

    seen, total_tokens, total_nll, batch = set(), 0, 0.0, 0
    scoring_start = time.monotonic()
    first_input, last_score, gpu_seconds = None, None, 0.0
    repeat_rows, repeat_scores = [], {}
    with (args.output_dir/'scores.jsonl').open('w') as stream:
        for line in sys.stdin:
            rows = json.loads(line)
            if first_input is None:
                first_input = time.monotonic()
            batch_start = time.monotonic()
            values = score(rows)
            last_score = time.monotonic()
            gpu_seconds += last_score-batch_start
            for row, value in zip(rows, values, strict=True):
                if value['id'] in seen:
                    raise ValueError('Duplicate document')
                seen.add(value['id'])
                total_tokens += value['scored_tokens']
                total_nll += value['nll']
                stream.write(json.dumps(value)+'\n')
                if len(repeat_rows) < 32:
                    repeat_rows.append(row)
                    repeat_scores[value['id']] = value
            stream.flush()
            batch += 1
            emit(phase='progress', batch=batch, documents=len(seen), scored_tokens=total_tokens,
                 batch_seconds=last_score-batch_start, elapsed_scoring_seconds=last_score-scoring_start,
                 cumulative_generate_seconds=gpu_seconds)
    if not seen:
        raise ValueError('No documents scored')
    corpus_finished = time.monotonic()
    checks = []
    for name, groups in [('reverse', [list(reversed(repeat_rows))]),
                         ('groups_of_four', [repeat_rows[i:i+4] for i in range(0, len(repeat_rows), 4)])]:
        repeated = [value for group in groups for value in score(group)]
        changed = [v['id'] for v in repeated if v != repeat_scores[v['id']]]
        checks.append({'case': name, 'documents': len(repeated),
                       'scored_tokens': sum(v['scored_tokens'] for v in repeated), 'different_documents': changed})
        emit(phase='repeatability', **checks[-1])
    report = {'role': args.role, 'documents': len(seen), 'scored_tokens': total_tokens,
              'load_seconds': load_seconds, 'configuration_validation_seconds': validation_seconds,
              'scoring_wall_seconds': last_score-first_input, 'generate_seconds': gpu_seconds,
              'first_input_monotonic': first_input, 'last_score_monotonic': last_score,
              'repeatability_seconds': time.monotonic()-corpus_finished,
              'mean_nll': total_nll/total_tokens, 'repeatability': checks}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    emit(phase='complete', **{k: v for k, v in report.items() if k != 'role'})
    if any(c['different_documents'] for c in checks):
        raise AssertionError('Token-loss repeatability failed')


if __name__ == '__main__':
    main()
