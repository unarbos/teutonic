"""Document isolation for the pinned MiMo model, without changing its source.

Eager uses explicit per-window masks. FA4 keeps the [batch, window] shape,
but supplies explicit cumulative fragment lengths to its varlen kernel. The
MiMo forward does not propagate arbitrary varlen kwargs to attention, so a
scoped adapter supplies them at the Transformers attention interface.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

from teutonic.evaluation.masking import DOCUMENT_EOS_TOKEN_ID


@dataclass
class DocumentLayout:
    positions: torch.Tensor
    valid_targets: torch.Tensor
    boundaries: torch.Tensor
    max_length: int
    devices: dict = field(default_factory=dict)
    visited_layers: set[int] = field(default_factory=set)

    def flash_kwargs(self, device):
        if device not in self.devices:
            cu = self.boundaries.to(device=device, dtype=torch.int32)
            self.devices[device] = {
                "cu_seq_lens_q": cu,
                "cu_seq_lens_k": cu,
                "max_length_q": self.max_length,
                "max_length_k": self.max_length,
            }
        return self.devices[device]


def document_layout(input_ids: torch.Tensor, sample_lengths: list[int] | None = None) -> DocumentLayout:
    if input_ids.ndim != 2 or input_ids.shape[1] < 2 or input_ids.shape[0] < 1:
        raise ValueError("masked evaluation expects a nonempty matrix of windows of length >= 2")
    batch, width = input_ids.shape
    starts = torch.ones_like(input_ids, dtype=torch.bool)
    starts[:, 1:] = input_ids[:, :-1] == DOCUMENT_EOS_TOKEN_ID
    if sample_lengths is not None:
        if batch != 1 or sum(sample_lengths) != width or any(n < 2 for n in sample_lengths):
            raise ValueError("packed sample lengths must match the input tensor")
        offsets = torch.tensor([0, *sample_lengths[:-1]], device=input_ids.device).cumsum(0)
        starts[0, offsets] = True
    indices = torch.arange(width, device=input_ids.device).expand(batch, -1)
    last_start = torch.where(starts, indices, 0).cummax(dim=1).values
    positions = indices - last_start
    boundaries = torch.cat(
        (
            starts.reshape(-1).nonzero().flatten(),
            torch.tensor([input_ids.numel()], device=input_ids.device),
        )
    ).to(torch.int32)
    return DocumentLayout(
        positions=positions,
        valid_targets=~starts[:, 1:],
        boundaries=boundaries,
        max_length=int(torch.diff(boundaries).max().item()),
    )


def eager_document_masks(layout: DocumentLayout, dtype, sliding_window: int | None):
    positions = layout.positions
    width = positions.shape[1]
    indices = torch.arange(width, device=positions.device)
    starts = indices - positions
    distance = indices[:, None] - indices[None, :]
    allowed = (starts[:, :, None] == starts[:, None, :]) & (distance >= 0)

    def additive(mask):
        return (
            torch.zeros(mask.shape, dtype=dtype, device=positions.device)
            .masked_fill(~mask, torch.finfo(dtype).min)
            .unsqueeze(1)
        )

    masks = {"full_attention": additive(allowed)}
    if sliding_window is not None:
        masks["sliding_window_attention"] = additive(allowed & (distance < sliding_window))
    return masks


_active_layout: ContextVar[DocumentLayout | None] = ContextVar("document_layout", default=None)


def install_document_flash_adapter():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["flash_attention_4"]
    if getattr(original, "_teutonic_document_masked", False):
        return

    def document_flash(module, query, key, value, attention_mask, **kwargs):
        layout = _active_layout.get()
        if layout is None:
            raise RuntimeError("FA4 evaluator attention requires an active document layout")
        expected = tuple(layout.positions.shape)
        if (query.shape[0], query.shape[2]) != expected or key.shape[2] != expected[1]:
            raise RuntimeError("document boundaries do not match attention tensor shapes")
        if attention_mask is not None:
            raise RuntimeError("packed document attention must not use the padding-mask path")
        kwargs.update(layout.flash_kwargs(query.device))
        layout.visited_layers.add(module.layer_idx)
        return original(module, query, key, value, None, **kwargs)

    document_flash._teutonic_document_masked = True
    document_flash._teutonic_document_backend = original
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_4", document_flash)


@contextmanager
def masked_model_inputs(model, input_ids, sample_lengths=None):
    if getattr(model.config, "model_type", None) != "mimo_v2":
        raise ValueError("document-masked scoring supports the pinned MiMo model only")
    layout = document_layout(input_ids, sample_lengths)
    implementation = model.config._attn_implementation
    if implementation == "flash_attention_4":
        install_document_flash_adapter()
        # Bypass Transformers' generic mask construction. Explicit cu_seqlens
        # enforce both EOS boundaries and boundaries between original windows.
        masks = {"full_attention": None, "sliding_window_attention": None}
    elif implementation == "eager":
        masks = eager_document_masks(
            layout, next(model.parameters()).dtype, getattr(model.config, "sliding_window", None)
        )
    else:
        raise ValueError(f"unsupported masked attention backend: {implementation}")
    token = _active_layout.set(layout)
    try:
        yield layout, {"position_ids": layout.positions, "attention_mask": masks}
        if (
            implementation == "flash_attention_4"
            and len(layout.visited_layers) != model.config.num_hidden_layers
        ):
            raise RuntimeError("not every model layer used document-isolated FA4 attention")
    finally:
        _active_layout.reset(token)
