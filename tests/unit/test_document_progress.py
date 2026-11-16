"""Progress survives engine normalization and the PostgreSQL progress allowlist."""

import ast
import threading
import time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

from teutonic.evaluator.document_progress import DocumentProgress
from teutonic.validator.repository import _bounded_progress


def test_streaming_progress_uses_tokens_until_final_document_count():
    events = []
    progress = DocumentProgress(0, events.append, token_budget=100)
    progress({'phase': 'dataset_loading'})
    assert events[-1]['percent'] == 0
    progress.selected(2)
    progress.reference_scored(2, input_tokens=40)
    assert events[-1]['percent'] == 20
    progress.selected(6)
    progress({'phase': 'sampling_documents'})
    assert events[-1]['percent'] == 20
    progress.reference_scored(6, input_tokens=101)
    assert events[-1]['percent'] == 49.9
    progress.finish_sampling(6)
    progress.reference_scored(6)
    assert events[-1]['percent'] == 50
    progress.begin_paired()
    progress.paired_scored(6, {})
    assert events[-1]['percent'] == 100
    assert [e['percent'] for e in events] == sorted(e['percent'] for e in events)


def test_two_stage_progress_and_lcb_survive_worker_events_and_heartbeats():
    # Run the production callback without importing the GPU model stack.
    path = Path(__file__).parents[2] / "teutonic/evaluator/engine.py"
    run_eval = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "run_eval")
    callback = next(n for n in run_eval.body if isinstance(n, ast.FunctionDef) and n.name == "on_phase")
    record = SimpleNamespace(progress=None, event=lambda kind, data: data)
    events = Queue()
    namespace = {"record": record, "events": events, "downloads": None,
                 "progress_lock": threading.RLock(), "t0": time.time(), "time": time}
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 -- trusted repository callback
    progress = DocumentProgress(8, namespace["on_phase"])
    progress.reference_scored(6)  # Includes cached documents and duplicate text.
    assert record.progress['percent'] == 37.5
    progress({"phase": "scoring", "role": "horizon", "documents": 1})
    assert record.progress['phase'] == 'reference_scoring'
    assert record.progress['completed_sequences'] == 6
    assert 'provisional_lcb' not in record.progress
    progress.reference_scored(8)
    progress.begin_paired()
    assert record.progress['percent'] == 50
    assert record.progress['completed_sequences'] == 0
    provisional = {"provisional_lcb": .001, "provisional_mu_hat": .002,
                   "provisional_n_sequences": 2, "provisional_n_bootstrap": 1000}
    progress.paired_scored(2, provisional)
    progress({"phase": "model_loading"})
    progress({"phase": "scoring", "role": "challenger", "documents": 1})
    namespace['on_phase']({**record.progress, 'heartbeat_at': time.time()})
    saved = _bounded_progress(record.progress)
    assert saved['percent'] == 62.5
    assert saved['phase'] == 'paired_scoring'
    assert saved['completed_sequences'] == 2
    assert all(saved[key] == value for key, value in provisional.items())
    assert 'role' not in saved
    progress.paired_scored(8, {**provisional, 'provisional_n_sequences': 8})
    assert record.progress['percent'] == 100
    percentages = []
    while not events.empty():
        percentages.append(events.get()['percent'])
    assert percentages == sorted(percentages)
