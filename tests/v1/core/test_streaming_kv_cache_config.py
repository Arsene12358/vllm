# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the streaming-kv config flags and SinkWindowSpec emission.

U7 — CacheConfig.streaming_kv_{start_size,recent_size} validator:
     - both unset      -> ok (default; streaming KV disabled)
     - both set, > 0   -> ok
     - one set, other not -> ValueError
     - either <= 0     -> ValueError
U8 — CLI: --streaming-kv-start-size / --streaming-kv-recent-size round-trip
     through EngineArgs.add_cli_args / from_cli_args.
U9 — Attention.get_kv_cache_spec emits SinkWindowSpec for decoder layers when
     the flags are set (taking precedence over the sliding-window branch),
     keeps the default spec when unset, and rejects MLA loudly (both the
     Attention assert and the MLAAttention layer class).
U10 — unify_hybrid_kv_cache_specs (disable-hybrid-kv-cache-manager path) never
      down-converts SinkWindowSpec: uniform sink-window specs pass through
      untouched, mixed specs raise a clear error instead of silently dropping
      the eviction feature.
G1..G4 — startup guards for configurations the SinkWindow port does not
      support. Each has a positive leg (the guard fires on the bad config) and
      a negative leg (the validated demo config — FA3 backend, no spec decode,
      no KV connector, standard Attention layers — is untouched):
      G1 attention backend must be FLASH_ATTN (only its metadata builder
         splices the evicted null hole out of the block-table row), checked
         both at spec emission and after metadata-builder creation;
      G2 speculative decoding is rejected at config time;
      G3 KV connectors are rejected at config time;
      G4 every decoder attention layer must end up with SinkWindowSpec (model
         classes that override get_kv_cache_spec never see the flags).
R1..R5 — the --streaming-kv-rebase-at config wall. Every check has both legs
      (fires on the bad config, inert on the good one and without the flag):
      R1 CacheConfig validator: rebase_at requires both streaming-kv flags,
         and rebase_at >= start_size + 2*recent_size (the single-rotation
         invariant), boundary tested both ways;
      R2 VllmConfig.validate_streaming_kv KV-dtype gate: the rebase rotation's
         write-back rounding bound is only validated for bf16/fp16 caches, so
         quantized KV dtypes — explicit, or "auto" resolving to a
         checkpoint-declared KV quantization or a non-bf16/fp16 model dtype —
         are rejected in rebase mode;
      R3 VllmConfig.validate_block_size: start_size % block_size == 0 in
         rebase mode, seated after the attention backend has finalised
         block_size (it may renegotiate the config-time value);
      R4 CLI: --streaming-kv-rebase-at round-trip;
      R5 VllmConfig.validate_streaming_kv fences on the other two M-RoPE
         position stores/writers the rebase seats do not reach: the V2 model
         runner (its own RopeState) and multimodal pruning (rewrites both
         fields after the seats).
"""

import argparse
from types import SimpleNamespace

import pytest
import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.cache import CacheConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.core.kv_cache_utils import (
    unify_hybrid_kv_cache_specs,
    verify_streaming_kv_specs_uniform,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SinkWindowSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    verify_sink_window_metadata_builders,
)

pytestmark = pytest.mark.cpu_test


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
class _FakeBackend:
    """Stand-in for a resolved attention backend class."""

    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name

    def is_mla(self) -> bool:
        return False


class _FakeAttentionLayer:
    """Minimal stand-in for Attention with the attributes get_kv_cache_spec
    reads. Building a real Attention requires a full ModelConfig + backend."""

    attn_type = AttentionType.DECODER
    kv_cache_dtype = "auto"
    kv_cache_torch_dtype = torch.bfloat16
    layer_name = "model.layers.0.self_attn.attn"
    num_kv_heads = 4
    head_size = 128
    head_size_v = 128
    sliding_window = None
    # The validated configuration resolves to FA3, whose builder is the only
    # one that compacts the block-table row.
    attn_backend = _FakeBackend("FLASH_ATTN")


def _make_vllm_config(start=None, recent=None, use_mla=False, block_size=16):
    cache_config = CacheConfig(
        block_size=block_size,
        streaming_kv_start_size=start,
        streaming_kv_recent_size=recent,
    )
    return SimpleNamespace(
        cache_config=cache_config,
        model_config=SimpleNamespace(use_mla=use_mla),
    )


def _make_sink_spec(start_size=64, sliding_window=512, num_kv_heads=4):
    return SinkWindowSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=sliding_window,
        start_size=start_size,
    )


# ----------------------------------------------------------------------------
# U7 — CacheConfig validator
# ----------------------------------------------------------------------------
def test_streaming_kv_disabled_by_default():
    c = CacheConfig()
    assert c.streaming_kv_start_size is None
    assert c.streaming_kv_recent_size is None


def test_streaming_kv_both_set_ok():
    c = CacheConfig(
        streaming_kv_start_size=64,
        streaming_kv_recent_size=8192,
    )
    assert c.streaming_kv_start_size == 64
    assert c.streaming_kv_recent_size == 8192


@pytest.mark.parametrize(
    "kwargs",
    [
        {"streaming_kv_start_size": 64},
        {"streaming_kv_recent_size": 8192},
    ],
)
def test_streaming_kv_partial_rejected(kwargs):
    with pytest.raises(ValueError, match="must be set together"):
        CacheConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"streaming_kv_start_size": 0, "streaming_kv_recent_size": 8192},
        {"streaming_kv_start_size": -1, "streaming_kv_recent_size": 8192},
        {"streaming_kv_start_size": 64, "streaming_kv_recent_size": 0},
        {"streaming_kv_start_size": 64, "streaming_kv_recent_size": -100},
    ],
)
def test_streaming_kv_nonpositive_rejected(kwargs):
    with pytest.raises(ValueError, match=r"must be > 0"):
        CacheConfig(**kwargs)


# ----------------------------------------------------------------------------
# U8 — CLI plumbing
# ----------------------------------------------------------------------------
def test_streaming_kv_cli_args_roundtrip():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--streaming-kv-start-size",
            "128",
            "--streaming-kv-recent-size",
            "1024",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.streaming_kv_start_size == 128
    assert engine_args.streaming_kv_recent_size == 1024


def test_streaming_kv_cli_defaults_to_none():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    engine_args = EngineArgs.from_cli_args(parser.parse_args([]))
    assert engine_args.streaming_kv_start_size is None
    assert engine_args.streaming_kv_recent_size is None


# ----------------------------------------------------------------------------
# U9 — SinkWindowSpec emission in Attention.get_kv_cache_spec
# ----------------------------------------------------------------------------
def test_get_kv_cache_spec_emits_sink_window_spec():
    cfg = _make_vllm_config(start=128, recent=1024)
    spec = Attention.get_kv_cache_spec(_FakeAttentionLayer(), cfg)
    assert isinstance(spec, SinkWindowSpec)
    assert spec.block_size == 16
    assert spec.num_kv_heads == 4
    assert spec.head_size == 128
    assert spec.dtype == torch.bfloat16
    assert spec.kv_quant_mode == get_kv_quant_mode("auto")
    assert spec.sliding_window == 1024
    assert spec.start_size == 128


def test_get_kv_cache_spec_streaming_overrides_sliding_window():
    """The streaming-kv branch must run before the per-layer sliding-window
    branch so every decoder layer emits SinkWindowSpec uniformly."""
    layer = _FakeAttentionLayer()
    layer.sliding_window = 4096
    cfg = _make_vllm_config(start=64, recent=512)
    spec = Attention.get_kv_cache_spec(layer, cfg)
    assert isinstance(spec, SinkWindowSpec)
    assert spec.sliding_window == 512
    assert spec.start_size == 64


def test_get_kv_cache_spec_default_path_unchanged_when_unset():
    cfg = _make_vllm_config()
    spec = Attention.get_kv_cache_spec(_FakeAttentionLayer(), cfg)
    assert isinstance(spec, FullAttentionSpec)
    assert not isinstance(spec, SinkWindowSpec)


def test_streaming_kv_mla_rejected():
    cfg = _make_vllm_config(start=128, recent=1024, use_mla=True)
    with pytest.raises(AssertionError, match="MLA is not supported for sink"):
        Attention.get_kv_cache_spec(_FakeAttentionLayer(), cfg)


def test_streaming_kv_mla_attention_layer_rejected():
    """MLA models route through MLAAttention.get_kv_cache_spec, which must
    fail loudly instead of silently ignoring the streaming-kv flags."""
    cfg = _make_vllm_config(start=128, recent=1024, use_mla=True)
    with pytest.raises(AssertionError, match="MLA is not supported for sink"):
        MLAAttention.get_kv_cache_spec(SimpleNamespace(), cfg)


# ----------------------------------------------------------------------------
# U10 — disable-hybrid down-conversion must never touch SinkWindowSpec
# ----------------------------------------------------------------------------
def test_unify_hybrid_specs_keeps_uniform_sink_window():
    specs = {f"model.layers.{i}.self_attn.attn": _make_sink_spec() for i in range(4)}
    unify_hybrid_kv_cache_specs(specs)
    assert all(isinstance(spec, SinkWindowSpec) for spec in specs.values())


def test_unify_hybrid_specs_rejects_mixed_sink_window():
    specs = {
        "model.layers.0.self_attn.attn": _make_sink_spec(),
        "model.layers.1.self_attn.attn": FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
        ),
    }
    with pytest.raises(ValueError, match="must not be down-converted"):
        unify_hybrid_kv_cache_specs(specs)


# ----------------------------------------------------------------------------
# G1 — the resolved attention backend must be FLASH_ATTN
# ----------------------------------------------------------------------------
def test_streaming_kv_requires_flash_attn_backend():
    """Only FlashAttentionMetadataBuilder splices the evicted null hole out of
    the block-table row; any other backend reads freed KV silently."""
    layer = _FakeAttentionLayer()
    layer.attn_backend = _FakeBackend("TRITON_ATTN")
    cfg = _make_vllm_config(start=128, recent=1024)
    with pytest.raises(AssertionError, match="FLASH_ATTN"):
        Attention.get_kv_cache_spec(layer, cfg)


def test_streaming_kv_backend_guard_inert_on_flash_attn():
    """Validated config: FA backend -> SinkWindowSpec, no guard fires."""
    cfg = _make_vllm_config(start=128, recent=1024)
    spec = Attention.get_kv_cache_spec(_FakeAttentionLayer(), cfg)
    assert isinstance(spec, SinkWindowSpec)


def test_backend_guard_does_not_fire_when_streaming_kv_unset():
    """A non-FA backend is perfectly fine when the flags are not set."""
    layer = _FakeAttentionLayer()
    layer.attn_backend = _FakeBackend("TRITON_ATTN")
    spec = Attention.get_kv_cache_spec(layer, _make_vllm_config())
    assert isinstance(spec, FullAttentionSpec)


def _make_attn_group(spec, builder):
    return AttentionGroup(
        backend=FlashAttentionBackend,
        layer_names=["model.layers.0.self_attn.attn"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        metadata_builders=[builder],
    )


def _make_flash_attn_builder(spec):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _pc: 4,
            get_num_kv_heads=lambda _pc: 4,
            get_head_size=lambda: 128,
            max_model_len=512,
            rswa_window=None,
        ),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False),
            max_cudagraph_capture_size=None,
        ),
        attention_config=SimpleNamespace(flash_attn_max_num_splits_for_cuda_graph=1),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    return FlashAttentionMetadataBuilder(
        spec, ["model.layers.0.self_attn.attn"], vllm_config, torch.device("cpu")
    )


def test_metadata_builder_guard_rejects_non_flash_attn_builder():
    """Second seat for G1: catches a backend swapped in after spec emission
    (e.g. via VLLM_ATTENTION_BACKEND or block-size renegotiation)."""

    class _OtherBuilder:
        pass

    group = _make_attn_group(_make_sink_spec(), _OtherBuilder())
    with pytest.raises(ValueError, match="FLASH_ATTN"):
        verify_sink_window_metadata_builders([group])


def test_metadata_builder_guard_inert_on_flash_attn():
    spec = _make_sink_spec(start_size=32, sliding_window=64)
    group = _make_attn_group(spec, _make_flash_attn_builder(spec))
    verify_sink_window_metadata_builders([group])  # must not raise


def test_metadata_builder_guard_ignores_non_sink_specs():
    """Groups without SinkWindowSpec are none of this guard's business."""

    class _OtherBuilder:
        pass

    full = FullAttentionSpec(
        block_size=16, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
    )
    verify_sink_window_metadata_builders([_make_attn_group(full, _OtherBuilder())])


# ----------------------------------------------------------------------------
# G2/G3 — config-time feature-incompatibility guards
# ----------------------------------------------------------------------------
_STREAMING_KV = {"streaming_kv_start_size": 64, "streaming_kv_recent_size": 1024}


def _ngram_spec_config():
    return SpeculativeConfig(
        method="ngram", num_speculative_tokens=3, prompt_lookup_max=4
    )


def test_streaming_kv_rejects_speculative_decoding():
    with pytest.raises(ValueError, match="not supported with speculative decoding"):
        VllmConfig(
            cache_config=CacheConfig(**_STREAMING_KV),
            speculative_config=_ngram_spec_config(),
        )


def test_speculative_decoding_allowed_without_streaming_kv():
    VllmConfig(speculative_config=_ngram_spec_config())


def test_streaming_kv_rejects_kv_connector():
    with pytest.raises(ValueError, match="not supported with KV connectors"):
        VllmConfig(
            cache_config=CacheConfig(**_STREAMING_KV),
            kv_transfer_config=KVTransferConfig(
                kv_connector="OffloadingConnector", kv_role="kv_both"
            ),
        )


def test_streaming_kv_rejects_kv_offloading_derived_connector():
    """--kv-offloading-size mints a connector in _post_init_kv_transfer_config,
    so the guard must read the resolved kv_transfer_config, not the CLI flag."""
    with pytest.raises(ValueError, match="not supported with KV connectors"):
        VllmConfig(cache_config=CacheConfig(kv_offloading_size=1, **_STREAMING_KV))


def test_kv_connector_allowed_without_streaming_kv():
    VllmConfig(
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector", kv_role="kv_both"
        )
    )


def test_validated_streaming_kv_config_passes():
    """The validated demo config: streaming-kv alone, no spec decode, no KV
    connector. Neither config-time guard may fire."""
    cfg = VllmConfig(cache_config=CacheConfig(**_STREAMING_KV))
    assert cfg.cache_config.streaming_kv_start_size == 64
    assert cfg.cache_config.streaming_kv_recent_size == 1024


# ----------------------------------------------------------------------------
# G4 — every decoder attention layer must end up with SinkWindowSpec
# ----------------------------------------------------------------------------
def _streaming_cache_config():
    return CacheConfig(**_STREAMING_KV)


def test_uniformity_guard_rejects_layer_that_ignored_the_flags():
    """Model classes with their own get_kv_cache_spec (e.g. DeepseekV4Attention)
    never consult the flags; partial eviction must fail loudly."""
    specs = {
        "model.layers.0.self_attn.attn": _make_sink_spec(),
        "model.layers.1.self_attn.attn": FullAttentionSpec(
            block_size=16, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
        ),
    }
    with pytest.raises(ValueError, match=r"model\.layers\.1\.self_attn\.attn"):
        verify_streaming_kv_specs_uniform(specs, _streaming_cache_config())


def test_uniformity_guard_inert_on_uniform_sink_window():
    specs = {f"model.layers.{i}.self_attn.attn": _make_sink_spec() for i in range(4)}
    verify_streaming_kv_specs_uniform(specs, _streaming_cache_config())


def test_uniformity_guard_ignores_non_attention_specs():
    """Mamba/linear-attention layers hold no attention KV, so they are not
    candidates for sink+window eviction."""
    specs = {
        "model.layers.0.self_attn.attn": _make_sink_spec(),
        "model.layers.1.mixer": MambaSpec(
            block_size=16, shapes=((4, 8),), dtypes=(torch.bfloat16,)
        ),
    }
    verify_streaming_kv_specs_uniform(specs, _streaming_cache_config())


def test_uniformity_guard_off_when_streaming_kv_unset():
    specs = {
        "model.layers.0.self_attn.attn": FullAttentionSpec(
            block_size=16, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
        ),
    }
    verify_streaming_kv_specs_uniform(specs, CacheConfig())


# ----------------------------------------------------------------------------
# R1 — CacheConfig validator: --streaming-kv-rebase-at arithmetic wall
# ----------------------------------------------------------------------------
# Production-shaped values: 128 + 2*1024 = 2176 <= 49152.
_REBASE_OK = {
    "streaming_kv_start_size": 128,
    "streaming_kv_recent_size": 1024,
    "streaming_kv_rebase_at": 49152,
}


def test_rebase_at_disabled_by_default():
    assert CacheConfig().streaming_kv_rebase_at is None
    assert CacheConfig(**_STREAMING_KV).streaming_kv_rebase_at is None


def test_rebase_at_valid_config_ok():
    c = CacheConfig(**_REBASE_OK)
    assert c.streaming_kv_rebase_at == 49152


def test_rebase_at_requires_streaming_kv_flags():
    with pytest.raises(
        ValueError,
        match=r"--streaming-kv-start-size and\s+--streaming-kv-recent-size",
    ):
        CacheConfig(streaming_kv_rebase_at=49152)


@pytest.mark.parametrize("rebase_at", [2175, 2112, 1, 0, -5])
def test_rebase_at_below_single_rotation_threshold_rejected(rebase_at):
    """128 + 2*1024 = 2176 is the minimum spacing that keeps consecutive
    rebase events one full recent window apart (single-rotation invariant)."""
    with pytest.raises(ValueError, match="single-rotation invariant"):
        CacheConfig(
            streaming_kv_start_size=128,
            streaming_kv_recent_size=1024,
            streaming_kv_rebase_at=rebase_at,
        )


def test_rebase_at_exactly_at_threshold_ok():
    """The invariant is >=, so the boundary itself must pass."""
    c = CacheConfig(
        streaming_kv_start_size=128,
        streaming_kv_recent_size=1024,
        streaming_kv_rebase_at=2176,
    )
    assert c.streaming_kv_rebase_at == 2176


def test_rebase_at_threshold_error_shows_arithmetic():
    with pytest.raises(ValueError, match=r"128 \+ 2\*1024 = 2176"):
        CacheConfig(
            streaming_kv_start_size=128,
            streaming_kv_recent_size=1024,
            streaming_kv_rebase_at=2000,
        )


# ----------------------------------------------------------------------------
# R2 — KV-dtype gate for rebase mode (VllmConfig.validate_streaming_kv seat)
# ----------------------------------------------------------------------------
def _fake_model_config(
    dtype=torch.bfloat16, quantization_config=None, multimodal_config=None
):
    """Just the attributes the rebase gates read; a real ModelConfig needs hub
    access. Mirrors resolve_kv_cache_dtype_string's hf_config traversal."""
    return SimpleNamespace(
        dtype=dtype,
        hf_config=SimpleNamespace(quantization_config=quantization_config),
        multimodal_config=multimodal_config,
    )


def _validate_streaming_kv(cache_config, model_config, use_v2_model_runner=False):
    fake = SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        speculative_config=None,
        kv_transfer_config=None,
        use_v2_model_runner=use_v2_model_runner,
    )
    return VllmConfig.validate_streaming_kv(fake)


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3", "fp8_e5m2", "nvfp4"])
def test_rebase_rejects_quantized_kv_cache_dtype(cache_dtype):
    with pytest.raises(ValueError, match="bf16 or fp16 KV cache"):
        VllmConfig(cache_config=CacheConfig(cache_dtype=cache_dtype, **_REBASE_OK))


@pytest.mark.parametrize("cache_dtype", ["bfloat16", "float16"])
def test_rebase_accepts_half_precision_kv_cache_dtype(cache_dtype):
    cfg = VllmConfig(cache_config=CacheConfig(cache_dtype=cache_dtype, **_REBASE_OK))
    assert cfg.cache_config.streaming_kv_rebase_at == 49152


def test_quantized_kv_cache_dtype_allowed_without_rebase():
    """The dtype gate is rebase-only: plain streaming-kv never rotates cached
    K, so fp8 stays as supported as it was before this flag existed."""
    cfg = VllmConfig(cache_config=CacheConfig(cache_dtype="fp8", **_STREAMING_KV))
    assert cfg.cache_config.cache_dtype == "fp8"


@pytest.mark.parametrize(
    "model_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"]
)
def test_rebase_auto_dtype_accepts_half_precision_model(model_dtype):
    cfg = _validate_streaming_kv(
        CacheConfig(**_REBASE_OK), _fake_model_config(dtype=model_dtype)
    )
    assert cfg.cache_config.streaming_kv_rebase_at == 49152


def test_rebase_auto_dtype_rejects_fp32_model():
    """ "auto" resolves to the model dtype; fp32 is outside the validated
    bf16/fp16 write-back envelope."""
    with pytest.raises(ValueError, match="float32"):
        _validate_streaming_kv(
            CacheConfig(**_REBASE_OK), _fake_model_config(dtype=torch.float32)
        )


def test_rebase_auto_dtype_rejects_fp8_kv_checkpoint():
    """Engine entry points fold checkpoint-declared KV quantization into
    cache_dtype before VllmConfig is built (resolve_kv_cache_dtype_string);
    the gate re-runs the sniff so directly-constructed configs cannot slip
    an fp8-KV checkpoint through as "auto"."""
    quant = {"quant_method": "modelopt", "kv_cache_quant_algo": "fp8"}
    with pytest.raises(ValueError, match="KV cache quantization"):
        _validate_streaming_kv(
            CacheConfig(**_REBASE_OK),
            _fake_model_config(dtype=torch.bfloat16, quantization_config=quant),
        )


def test_rebase_auto_dtype_rejects_compressed_tensors_kv_scheme():
    """compressed-tensors FP8-KV checkpoints (RedHatAI style) sit outside the
    modelopt-only sniff, so nothing rewrites cache_dtype before model load —
    where Attention.__init__ flips "auto" to fp8 AFTER every validator has
    run. The gate mirrors that trigger (non-null kv_cache_scheme) and rejects
    up front."""
    quant = {
        "quant_method": "compressed-tensors",
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    }
    with pytest.raises(ValueError, match="kv_cache_scheme"):
        _validate_streaming_kv(
            CacheConfig(**_REBASE_OK),
            _fake_model_config(dtype=torch.bfloat16, quantization_config=quant),
        )


def test_rebase_auto_dtype_accepts_compressed_tensors_weight_only():
    """Weight-only compressed-tensors checkpoints carry kv_cache_scheme: null
    in config.json; the KV cache stays model-dtype and rebase stays allowed."""
    quant = {"quant_method": "compressed-tensors", "kv_cache_scheme": None}
    cfg = _validate_streaming_kv(
        CacheConfig(**_REBASE_OK),
        _fake_model_config(dtype=torch.bfloat16, quantization_config=quant),
    )
    assert cfg.cache_config.streaming_kv_rebase_at == 49152


def test_rebase_auto_dtype_without_model_config_skips():
    """model_config is only None in unit tests; every engine entry point
    builds one before VllmConfig. The explicit-string leg still applies."""
    cfg = VllmConfig(cache_config=CacheConfig(**_REBASE_OK))
    assert cfg.cache_config.streaming_kv_rebase_at == 49152


# ----------------------------------------------------------------------------
# R5 — rebase-only fences on the two other M-RoPE position stores/writers
# ----------------------------------------------------------------------------
def test_rebase_rejects_v2_model_runner():
    """The V2 runner keeps positions in its own RopeState, which neither
    rebase seat updates. (Unbound-call idiom as above: forcing the real
    `use_v2_model_runner` property on needs Triton, which CPU CI lacks.)"""
    with pytest.raises(ValueError, match="V2 model runner"):
        _validate_streaming_kv(
            CacheConfig(**_REBASE_OK), _fake_model_config(), use_v2_model_runner=True
        )


def test_v2_model_runner_allowed_without_rebase():
    """The V2 fence is rebase-only; plain streaming-kv is unchanged by it."""
    cfg = _validate_streaming_kv(
        CacheConfig(**_STREAMING_KV), _fake_model_config(), use_v2_model_runner=True
    )
    assert cfg.cache_config.streaming_kv_rebase_at is None


def test_rebase_rejects_multimodal_pruning():
    """EVS pruning rewrites both M-RoPE fields after the rebase seats."""
    mm_config = MultiModalConfig(video_pruning_rate=0.5)
    with pytest.raises(ValueError, match="multimodal pruning"):
        _validate_streaming_kv(
            CacheConfig(**_REBASE_OK),
            _fake_model_config(multimodal_config=mm_config),
        )


@pytest.mark.parametrize("rate", [None, 0.0])
def test_rebase_accepts_disabled_multimodal_pruning(rate):
    cfg = _validate_streaming_kv(
        CacheConfig(**_REBASE_OK),
        _fake_model_config(multimodal_config=MultiModalConfig(video_pruning_rate=rate)),
    )
    assert cfg.cache_config.streaming_kv_rebase_at == 49152


def test_multimodal_pruning_allowed_without_rebase():
    cfg = _validate_streaming_kv(
        CacheConfig(**_STREAMING_KV),
        _fake_model_config(multimodal_config=MultiModalConfig(video_pruning_rate=0.5)),
    )
    assert cfg.cache_config.streaming_kv_rebase_at is None


# ----------------------------------------------------------------------------
# R3 — sink-block alignment gate (VllmConfig.validate_block_size seat)
# ----------------------------------------------------------------------------
def test_rebase_unaligned_start_size_rejected_at_block_size_seat():
    """Rotation starts at the sink-block boundary (sinks are packed as whole
    blocks) while the rebase lands the window front at start_size; unaligned
    start_size would leave the pinned slack tokens [start_size, boundary)
    overlapped by rebased positions but never rotated."""
    cfg = VllmConfig(
        cache_config=CacheConfig(
            streaming_kv_start_size=120,
            streaming_kv_recent_size=1024,
            streaming_kv_rebase_at=49152,
        )
    )
    cfg.cache_config.block_size = 16
    with pytest.raises(ValueError, match="multiple of"):
        cfg.validate_block_size()


def test_rebase_alignment_checked_against_renegotiated_block_size():
    """Why this seat and not the config validator: the attention backend may
    renegotiate block_size after config validation (Platform.
    update_block_size_for_backend), and validate_block_size is the documented
    post-finalisation check point (engine core calls it before KV init)."""
    cfg = VllmConfig(cache_config=CacheConfig(**_REBASE_OK))
    cfg.cache_config.block_size = 32  # renegotiated, still aligned: 128 % 32
    cfg.validate_block_size()
    cfg.cache_config.block_size = 48  # renegotiated, misaligned: 128 % 48
    with pytest.raises(ValueError, match="multiple of"):
        cfg.validate_block_size()


def test_rebase_aligned_start_size_passes_block_size_seat():
    cfg = VllmConfig(cache_config=CacheConfig(**_REBASE_OK))
    cfg.cache_config.block_size = 16
    cfg.validate_block_size()  # must not raise


def test_unaligned_start_size_allowed_without_rebase():
    """Plain streaming-kv keeps supporting unaligned start_size: the compacted
    row pins the slack tokens and nothing ever rotates them."""
    cfg = VllmConfig(
        cache_config=CacheConfig(
            streaming_kv_start_size=120, streaming_kv_recent_size=1024
        )
    )
    cfg.cache_config.block_size = 16
    cfg.validate_block_size()  # must not raise


# ----------------------------------------------------------------------------
# R4 — CLI plumbing for --streaming-kv-rebase-at
# ----------------------------------------------------------------------------
def test_streaming_kv_rebase_at_cli_roundtrip():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--streaming-kv-start-size",
            "128",
            "--streaming-kv-recent-size",
            "1024",
            "--streaming-kv-rebase-at",
            "49152",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.streaming_kv_start_size == 128
    assert engine_args.streaming_kv_recent_size == 1024
    assert engine_args.streaming_kv_rebase_at == 49152


def test_streaming_kv_rebase_at_cli_defaults_to_none():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    engine_args = EngineArgs.from_cli_args(parser.parse_args([]))
    assert engine_args.streaming_kv_rebase_at is None
