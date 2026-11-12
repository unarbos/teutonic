"""Local-fixture GPU diagnostic for Horizon batch-shape numerical differences."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from teutonic.evaluator import horizon_pipeline as pipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(pipeline.deterministic_env())
    import torch
    fixture = json.loads(args.fixture.read_text())
    rows = [{'id': f'doc-{i:03d}', 'tokens': fixture[i % len(fixture)]['tokens']} for i in range(128)]
    target = rows[0]
    length = len(target['tokens'])
    expected = np.load(args.baseline)['doc-000-repeat-0']
    report = {'runtime': pipeline.identity(), 'cases': []}
    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
    model = pipeline.HorizonPipeline(args.snapshot)
    baseline_hidden = {}
    captured = {}
    hooks = []
    tracing = True
    def hook(name):
        def record(module, inputs, output):
            if not tracing:
                return
            output = output[0] if isinstance(output, tuple) else output
            selected = output[0, :length].float().cpu().numpy().copy()
            if name not in baseline_hidden:
                baseline_hidden[name] = selected
            diff = selected - baseline_hidden[name]
            captured[name] = {'different': int(np.count_nonzero(diff)), 'max_abs': float(np.abs(diff).max())}
        return record
    for i, layer in enumerate(model.model.model.layers):
        for name in ('self_attn', 'mlp'):
            hooks.append(getattr(layer, name).register_forward_hook(hook(f'{i}.{name}')))
        if i == 0:
            for name, module in layer.self_attn.named_modules():
                if isinstance(module, torch.nn.Linear):
                    hooks.append(module.register_forward_hook(hook(f'0.self_attn.{name}')))
    original_reduce = pipeline.reduce_document_losses
    losses = {}
    def reduce(values, batch):
        offset = 0
        for row in batch:
            count = len(row['tokens'])-1
            losses[row['id']] = values[offset:offset+count].copy()
            offset += count
        return original_reduce(values, batch)
    pipeline.reduce_document_losses = reduce
    def run(name, group):
        captured.clear()
        losses.clear()
        started = time.monotonic()
        result = model.score(group)
        delta = losses[target['id']]-expected
        case = {'name':name, 'seconds':time.monotonic()-started,
                'different_tokens':int(np.count_nonzero(delta)), 'max_abs_token_delta':float(np.abs(delta).max()),
                'document_nll_delta':math.fsum(losses[target['id']].tolist())-math.fsum(expected.tolist()),
                'first_differing_components':[k for k,v in captured.items() if v['different']][:8],
                'components':dict(captured)}
        np.save(args.output/f'{name}.npy',losses[target['id']])
        report['cases'].append(case)
        save()
        print(json.dumps({k:v for k,v in case.items() if k!='components'}),flush=True)
        return result
    neighbors=[r for r in rows[1:] if length <= len(r['tokens']) <= 5000]
    run('solo',[target])
    run('solo-repeat',[target])
    run('pair-short',[target,neighbors[0]])
    run('pair-long',[target,max(neighbors,key=lambda r:len(r['tokens']))])
    run('many',[target,*neighbors[:15]])
    tracing=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    run('no-reduced-solo',[target])
    run('no-reduced-many',[target,*neighbors[:15]])
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    real_numpy=pipeline.np
    class PaddedNumpy:
        def __init__(self, multiple):
            self.multiple=multiple
        def __getattr__(self,name):
            return getattr(real_numpy,name)
        def zeros(self,shape,**kwargs):
            if len(shape)==2:
                shape=(shape[0],math.ceil(shape[1]/self.multiple)*self.multiple)
            return real_numpy.zeros(shape,**kwargs)
    for multiple in [128,256,1024,8192]:
        pipeline.np=PaddedNumpy(multiple)
        run(f'pad-{multiple}-solo',[target])
        run(f'pad-{multiple}-many',[target,*neighbors[:15]])
    pipeline.np=real_numpy
    for handle in hooks:
        handle.remove()
    pipeline.reduce_document_losses=original_reduce
    large=[{'id':f'large-{i}', 'tokens':rows[i%128]['tokens']} for i in range(1024)]
    started=time.monotonic()
    result=model.score(large)
    elapsed=time.monotonic()-started
    report['large']={'seconds':elapsed,'scored_tokens':sum(r['scored_tokens'] for r in result),
                     'scored_tokens_per_second':sum(r['scored_tokens'] for r in result)/elapsed}
    (args.output/'large.json').write_text(json.dumps(result))
    report['completed']=True
    save()


if __name__=='__main__':
    main()
