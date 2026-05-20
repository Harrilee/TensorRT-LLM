"""Phase detection helper for moe_trace_logger instrumentation.

The PyTorch flow in TensorRT-LLM exposes per-step `AttentionMetadata` via a
thread-local dict accessed through `get_model_extra_attrs()` (set by
`PyExecutor.model_forward` in `tensorrt_llm/_torch/pyexecutor/model_engine.py`).
Each AttentionMetadata carries `num_contexts` (context-phase sequences) and
`num_generations` (generation-phase sequences); see
`tensorrt_llm/_torch/attention_backend/interface.py`.

The phase classification is:
  - num_contexts  > 0 and num_generations == 0 -> "prefill"
  - num_contexts == 0 and num_generations  > 0 -> "decode"
  - both > 0 (chunked prefill / mixed batch)    -> "mixed"
  - otherwise (unknown / metadata missing)      -> "unknown"

Callers can pass an explicit AttentionMetadata, or rely on the thread-local
`get_model_extra_attrs()["attention_metadata"]` weakref.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["phase_of", "current_phase"]


def phase_of(attn_metadata: Any) -> str:
    """Classify a PyTorch-flow AttentionMetadata as prefill/decode/mixed.

    Returns "unknown" if metadata is None or lacks the expected fields.
    """
    if attn_metadata is None:
        return "unknown"
    num_contexts = getattr(attn_metadata, "num_contexts", None)
    num_generations = getattr(attn_metadata, "num_generations", None)
    if num_contexts is None or num_generations is None:
        return "unknown"
    if num_contexts > 0 and num_generations == 0:
        return "prefill"
    if num_contexts == 0 and num_generations > 0:
        return "decode"
    if num_contexts > 0 and num_generations > 0:
        return "mixed"
    return "unknown"


def current_phase() -> str:
    """Look up the active AttentionMetadata from the thread-local model
    extra-attrs dict and classify its phase.

    Mirrors how attention backends locate the metadata via
    `tensorrt_llm._torch.utils.get_model_extra_attrs`. Falls back to
    `get_global_attrs().attention_metadata` if extra-attrs is not active.

    Returns "unknown" if neither source has live metadata. Imports are
    deferred so this helper is importable from non-PyTorch contexts.
    """
    attn_md = _resolve_attn_metadata()
    return phase_of(attn_md)


def _resolve_attn_metadata() -> Optional[Any]:
    try:
        from tensorrt_llm._torch.utils import (get_global_attrs,
                                               get_model_extra_attrs)
    except Exception:
        return None

    attrs = get_model_extra_attrs()
    if attrs is not None:
        ref = attrs.get("attention_metadata")
        md = ref() if callable(ref) else ref
        if md is not None:
            return md

    try:
        ref = getattr(get_global_attrs(), "attention_metadata", None)
    except Exception:
        ref = None
    if ref is None:
        return None
    md = ref() if callable(ref) else ref
    return md
