"""Horizon pipeline accounting, batching and isolated process lifecycle."""

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from teutonic.evaluator import document_duel, vllm_runtime
from teutonic.evaluator import horizon_pipeline as scorer
from teutonic.evaluator import horizon_pipeline_runtime as runtime


def row(name, length):
    return {'id':name,'tokens':[3]*(length-1)+[1]}


def test_pipeline_batches_bound_padding_and_preserve_whole_long_documents():
    settings={**scorer.settings(),'batch_tokens':16,'max_batch':3}
    rows=[row(str(i),n) for i,n in enumerate([5,20,2,5,3,2])]
    groups=scorer.microbatches(rows,settings)
    assert [len(r['tokens']) for r in groups[0]] == [20]
    assert sorted(r['id'] for group in groups for r in group)==sorted(r['id'] for r in rows)
    for group in groups:
        assert len(group)<=3
        assert len(group)==1 or len(group)*max(len(r['tokens']) for r in group)<=16
    assert [r['id'] for group in groups for r in group if len(r['tokens'])==5]==['0','3']
    assert scorer.layer_counts(61)==[8,8,8,8,8,9,9,3]


def test_native_eos_context_and_total_request_bounds():
    settings={**scorer.settings(),'max_model_len':8,'max_request_tokens':9,'max_request_documents':3}
    scorer.validate_rows([{'id':'a','tokens':[3,1,4,1]},row('b',5)],settings,10)
    for rows in ([row('a',8)],[row('a',5),row('b',5)],[row('a',2),row('a',2)],
                 [{'id':'a','tokens':[2,3]}],[{'id':'a','tokens':[10,1]}]):
        with pytest.raises(ValueError):
            scorer.validate_rows(rows,settings,10)


def test_small_microbatches_pad_total_positions_without_changing_targets():
    configuration = scorer.settings()
    for count in [1, 2, 3, 8, 16]:
        rows = [row(str(i), 707) for i in range(count)]
        length = scorer.padded_length(rows, configuration)
        assert length * count >= 8192
        assert length == max(707, math.ceil(8192/count))
        assert [len(r['tokens'])-1 for r in rows] == [706] * count
    rows = [row(str(i), 12000) for i in range(8)]
    assert scorer.padded_length(rows, configuration) == 12000


def test_document_sums_independent_of_neighbors_and_head_boundaries():
    rows=[row('a',5),row('b',3)]
    losses=np.asarray([1e8,.25,1e-5,3,2,4],dtype=np.float32)
    together=scorer.reduce_document_losses(losses,rows)
    separately=scorer.reduce_document_losses(losses[:4],rows[:1])+scorer.reduce_document_losses(losses[4:],rows[1:])
    assert together==separately
    assert together[0]['nll']==math.fsum(losses[:4].tolist())
    assert together[0]['scored_tokens']==4
    assert together[0]['input_sha256']==scorer.input_hash(rows[0]['tokens'])
    with pytest.raises(ValueError):
        scorer.reduce_document_losses(losses[:-1],rows)
    with pytest.raises(ValueError):
        scorer.reduce_document_losses(np.asarray([float('nan')]),[row('a',2)])


def test_reference_token_windows_preserve_order_and_all_documents(monkeypatch):
    docs=[SimpleNamespace(doc_id=str(i)) for i in range(4)]
    sequences=[[3]*n for n in [4,5,7,3]]
    monkeypatch.setattr(document_duel,'prepared_batches',lambda *a,**kw:iter([(docs,sequences)]))
    monkeypatch.setattr(document_duel,'MAX_REQUEST_TOKENS',9)
    batches=list(document_duel.reference_batches(None,docs,None))
    assert [[d.doc_id for d in batch] for batch,_ in batches]==[['0','1'],['2'],['3']]
    assert [t for _,tokens in batches for t in tokens]==sequences


def test_worker_environment_pins_determinism_and_never_forwards_credentials(tmp_path,monkeypatch):
    monkeypatch.setenv('TEUTONIC_HORIZON_KERNEL_CACHE_DIR',str(tmp_path/'kernels'))
    monkeypatch.setenv('TEUTONIC_EVALUATOR_R2_SECRET_ACCESS_KEY','never-forward')
    monkeypatch.setenv('HF_TOKEN','never-forward')
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF','conflicting-value')
    env=runtime.worker_env(range(8))
    assert 'never-forward' not in env.values()
    assert 'PYTORCH_CUDA_ALLOC_CONF' not in env
    assert env['PYTORCH_ALLOC_CONF']=='expandable_segments:True'
    assert env['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert env['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'


def test_horizon_uses_shared_rpc_accounting_registry_and_cleanup(tmp_path,monkeypatch):
    source=tmp_path/'fake_horizon.py'
    source.write_text('''import argparse,json,os,sys
from teutonic.evaluator.horizon_pipeline import settings,input_hash
p=argparse.ArgumentParser();p.add_argument('--response-fd',type=int);a=p.parse_args()
with os.fdopen(a.response_fd,'w',buffering=1) as out:
 out.write(json.dumps({'type':'ready','identity':{'backend':'fixture'},'settings':settings(),'load_seconds':.1})+'\\n')
 for line in sys.stdin:
  msg=json.loads(line)
  if msg['type']=='stop':break
  rows=[{'id':r['id'],'nll':1.,'scored_tokens':len(r['tokens'])-1,'input_sha256':input_hash(r['tokens'])} for r in msg['rows']]
  out.write(json.dumps({'type':'scores','batch':msg['batch'],'seconds':.01,'rows':rows})+'\\n')
''')
    monkeypatch.setenv('TEUTONIC_HORIZON_KERNEL_CACHE_DIR',str(tmp_path/'kernels'))
    monkeypatch.setattr(runtime.HorizonWorker,'_command',lambda self,role,snapshot,fd,ids:[sys.executable,str(source),'--response-fd',str(fd)])
    worker=runtime.HorizonWorker('horizon',tmp_path,list(range(8)),tmp_path,{'backend':'fixture'},lambda _:None)
    try:
        assert any(w['role']=='horizon' for w in vllm_runtime.worker_status()['workers'])
        values=worker.score([SimpleNamespace(doc_id='a')],[[3,1]])
        assert values[0]['nll']==1. and worker.report['documents']==1
    finally:
        vllm_runtime.close_all_workers()
    assert worker.closed and worker.process.poll() is not None
