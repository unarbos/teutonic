"""Boundary-layout and FA4 adapter checks.

Real-model GPU validation uses cached checkpoints through the normal eight-GPU
replay in scripts/replay_cached_evaluation.py; these tests construct no models.
"""

import types

import pytest
import torch

from teutonic.evaluation.masking import DOCUMENT_EOS_TOKEN_ID as EOS
from teutonic.evaluator.document_masking import (
    document_layout,
    eager_document_masks,
    install_document_flash_adapter,
    masked_model_inputs,
)


def test_layout_covers_consecutive_eos_and_window_boundaries():
    ids = torch.tensor([[1, EOS, EOS, 2, 3, EOS], [4, 5, 6, 7, 8, 9]])
    layout = document_layout(ids)
    assert layout.positions.tolist() == [[0, 1, 0, 0, 1, 2], [0, 1, 2, 3, 4, 5]]
    assert layout.boundaries.tolist() == [0, 2, 3, 6, 12]
    assert layout.valid_targets.tolist() == [[True, False, False, True, True], [True] * 5]
    assert layout.max_length == 6
    masks = eager_document_masks(layout, torch.float32, 2)
    full = masks["full_attention"][0, 0] == 0
    assert full[4].tolist() == [False, False, False, True, True, False]
    assert full[2].tolist() == [False, False, True, False, False, False]
    assert (masks["sliding_window_attention"][1, 0, 5] == 0).tolist() == [False] * 4 + [True] * 2


def test_flash_adapter_supplies_explicit_boundaries_for_multiline_batches(monkeypatch):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    calls = []

    def backend(module, query, key, value, attention_mask, **kwargs):
        calls.append(kwargs)
        return query, None

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "flash_attention_4", backend)
    install_document_flash_adapter()
    adapter = ALL_ATTENTION_FUNCTIONS["flash_attention_4"]
    install_document_flash_adapter()
    assert ALL_ATTENTION_FUNCTIONS["flash_attention_4"] is adapter
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            model_type="mimo_v2",
            _attn_implementation="flash_attention_4",
            num_hidden_layers=2,
        )
    )
    ids = torch.tensor([[1, EOS, 2, 3], [4, 5, 6, 7]])
    query = torch.zeros(2, 2, 4, 8)
    with masked_model_inputs(model, ids):
        for layer in range(2):
            adapter(
                types.SimpleNamespace(layer_idx=layer),
                query,
                query,
                query,
                None,
                sliding_window=4,
                s_aux="sink",
                is_causal=True,
            )
    assert len(calls) == 2
    assert calls[0]["cu_seq_lens_q"].tolist() == [0, 2, 4, 8]
    assert calls[0]["cu_seq_lens_k"].dtype == torch.int32
    assert calls[0]["max_length_q"] == calls[0]["max_length_k"] == 4
    assert calls[0]["sliding_window"] == 4
    assert calls[0]["s_aux"] == "sink"
    with pytest.raises(RuntimeError, match="active document layout"):
        adapter(types.SimpleNamespace(layer_idx=0), query, query, query, None)
    with (
        pytest.raises(RuntimeError, match="not every model layer"),
        masked_model_inputs(model, ids),
    ):
        pass


def test_packed_samples_reset_positions_even_without_terminal_eos():
    # A normal window can end mid-document. Packing must still isolate the next sample.
    ids = torch.tensor([[1, 2, 3, 4, 5, EOS, 6, 7, 8]])
    layout = document_layout(ids, [3, 6])
    assert layout.positions.tolist() == [[0, 1, 2, 0, 1, 2, 0, 1, 2]]
    assert layout.boundaries.tolist() == [0, 3, 6, 9]
    assert layout.valid_targets.tolist() == [[True, True, False, True, True, False, True, True]]
    allowed = eager_document_masks(layout, torch.float32, None)["full_attention"][0, 0] == 0
    assert not allowed[3:, :3].any()
    assert allowed[5, 3:6].all()
    with pytest.raises(ValueError, match="match"):
        document_layout(ids, [4, 6])


def test_packed_loss_reduction_matches_separately_scored_samples():
    from teutonic.evaluation.masking import scored_token_counts
    from teutonic.evaluator.engine import sum_sample_losses

    rows = [[1, 2, 3], [4, 5, EOS, 6, 7, 8], [9, EOS]]
    lengths = [len(row) for row in rows]
    ids = torch.tensor([[token for row in rows for token in row]])
    layout = document_layout(ids, lengths)
    # Distinct fixed losses expose offset errors and accidental boundary targets.
    losses = torch.arange(1, ids.numel(), dtype=torch.float32).reshape(1, -1)
    losses = losses.masked_fill(~layout.valid_targets, 0)
    sums = sum_sample_losses(losses, lengths)
    assert sums.tolist() == [1 + 2, 4 + 5 + 7 + 8, 10]
    assert scored_token_counts(rows) == [2, 4, 1]
    assert (sums / torch.tensor(scored_token_counts(rows))).tolist() == [1.5, 6.0, 10.0]
