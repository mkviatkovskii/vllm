# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from tests.v1.attention.test_mla_backends import (
    create_common_attn_metadata,
    create_and_prepopulate_kv_cache,
    create_vllm_config,
    run_attention_backend,
    BatchSpec,
    _convert_dtype_to_torch,
    MockSparseMLAAttentionLayer,
)
from vllm.config.vllm import set_current_vllm_config
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.platforms import current_platform

from unittest.mock import MagicMock
import vllm.distributed.parallel_state as parallel_state

# Mock TP and DCP groups to avoid initialization errors in testing
parallel_state._TP = MagicMock()
parallel_state._TP.rank_in_group = 0
parallel_state._TP.world_size = 1

parallel_state._DCP = MagicMock()
parallel_state._DCP.rank_in_group = 0
parallel_state._DCP.world_size = 1

class MockIndexer:
    def __init__(self, topk_indices_buffer: torch.Tensor):
        self.topk_indices_buffer = topk_indices_buffer

@pytest.mark.parametrize("batch_spec_name", ["small_decode", "medium_decode"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_triton_mla_sparse_parity(
    batch_spec_name: str,
    kv_cache_dtype: str,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda:0")
    device_capability = current_platform.get_device_capability()

    # 1. Setup Model Parameters (DeepSeek-V2 style)
    model = "deepseek-ai/DeepSeek-V2"
    hf_config_override = {
        "index_topk": 2048,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "num_attention_heads": 128,
    }
    
    BATCH_SPECS = {
        "small_decode": BatchSpec(seq_lens=[1024, 1024], query_lens=[1, 1]),
        "medium_decode": BatchSpec(seq_lens=[2048, 4096], query_lens=[1, 1]),
    }
    batch_spec = BATCH_SPECS[batch_spec_name]
    block_size = 64 # Required by some sparse kernels (FlashMLA)
    
    num_gpu_blocks = sum((s + block_size - 1) // block_size for s in batch_spec.seq_lens) + 10
    
    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        hf_config_override=hf_config_override,
    )
    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    
    num_q_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    dtype = _convert_dtype_to_torch(vllm_config.model_config.dtype)
    kv_lora_rank = hf_config_override["kv_lora_rank"]
    qk_nope_head_dim = hf_config_override["qk_nope_head_dim"]
    qk_rope_head_dim = hf_config_override["qk_rope_head_dim"]
    v_head_dim = qk_nope_head_dim
    
    # 2. Generate Data
    common_attn_metadata = create_common_attn_metadata(batch_spec, block_size, device)
    
    query_vllm = torch.randn(
        batch_spec.batch_size, num_q_heads, qk_nope_head_dim + qk_rope_head_dim,
        dtype=dtype, device=device
    )
    
    kv_c_contexts = [torch.randn(s - q, kv_lora_rank, dtype=dtype, device=device) 
                     for s, q in zip(batch_spec.seq_lens, batch_spec.query_lens)]
    k_pe_contexts = [torch.randn(s - q, 1, qk_rope_head_dim, dtype=dtype, device=device)
                      for s, q in zip(batch_spec.seq_lens, batch_spec.query_lens)]
    
    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype,
    )

    # 3. Mock Indexer (Dense fallback for parity check)
    topk_tokens = vllm_config.model_config.hf_config.index_topk
    topk_indices_buffer = torch.full(
        (batch_spec.batch_size, topk_tokens), -1, dtype=torch.int32, device=device
    )
    for i, s_len in enumerate(batch_spec.seq_lens):
        indices = torch.arange(min(s_len, topk_tokens), dtype=torch.int32, device=device)
        topk_indices_buffer[i, :len(indices)] = indices
    mock_indexer = MockIndexer(topk_indices_buffer)

    # 4. Define and Filter Backends to Test
    sparse_backends = [
        AttentionBackendEnum.TRITON_MLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
    ]
    
    supported_backends = []
    with set_current_vllm_config(vllm_config):
        for backend_enum in sparse_backends:
            try:
                backend_cls = backend_enum.get_class()
                invalid_reasons = backend_cls.validate_configuration(
                    head_size=head_size,
                    dtype=dtype,
                    kv_cache_dtype=kv_cache_dtype,
                    block_size=block_size,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=True,
                    use_mm_prefix=False,
                    use_per_head_quant_scales=False,
                    device_capability=device_capability,
                    attn_type="decoder",
                )
                if not invalid_reasons:
                    supported_backends.append(backend_enum)
            except (ImportError, AttributeError):
                continue

    if not supported_backends:
        pytest.skip("No sparse MLA backends supported on this hardware.")

    # 5. Get Dense Reference Output
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_q_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
        disable_tp=True,
    ).to(device=device, dtype=dtype)
    torch.nn.init.normal_(mock_kv_b_proj.weight)

    kv_cache_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=vllm_config.model_config.dtype,
        cache_dtype_str=kv_cache_dtype,
    )

    dense_output = run_attention_backend(
        AttentionBackendEnum.TRITON_MLA,
        kv_cache_spec,
        ["layer1"],
        vllm_config,
        device,
        common_attn_metadata,
        query_vllm,
        torch.zeros(batch_spec.batch_size, kv_lora_rank, dtype=dtype, device=device),
        torch.zeros(batch_spec.batch_size, 1, qk_rope_head_dim, dtype=dtype, device=device),
        kv_cache,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        mock_kv_b_proj,
        q_scale=1.0,
        k_scale=1.0,
        kv_cache_dtype=kv_cache_dtype,
    )

    # 6. Run and Compare Supported Sparse Backends
    outputs = {"DENSE": dense_output}
    
    for backend_enum in supported_backends:
        backend_cls = backend_enum.get_class()
        impl_cls = backend_cls.get_impl_cls()
        builder_cls = backend_cls.get_builder_cls()
        
        with set_current_vllm_config(vllm_config):
            impl = impl_cls(
                num_heads=num_q_heads,
                head_size=head_size,
                scale=head_size**-0.5,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype=kv_cache_dtype,
                logits_soft_cap=None,
                attn_type="decoder",
                kv_sharing_target_layer_name=None,
                q_lora_rank=None,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_b_proj=mock_kv_b_proj,
                indexer=mock_indexer,
            )
            # Process weights if backend needs it (like Triton scaling)
            if hasattr(impl, "process_weights_after_loading"):
                impl.process_weights_after_loading(dtype)
            
            mock_layer = MockSparseMLAAttentionLayer(
                impl=impl,
                num_heads=num_q_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                device=device,
                W_UK=impl.W_UK if hasattr(impl, "W_UK") else torch.randn(kv_lora_rank, num_q_heads, qk_nope_head_dim, device=device, dtype=dtype),
                W_UV=impl.W_UV_raw if hasattr(impl, "W_UV_raw") else torch.randn(kv_lora_rank, num_q_heads, v_head_dim, device=device, dtype=dtype),
                q_scale=1.0,
                k_scale=1.0,
            )
            
            builder = builder_cls(kv_cache_spec, ["layer1"], vllm_config, device)
            attn_metadata = builder.build(0, common_attn_metadata)
            
            sparse_out = torch.empty_like(dense_output)
            sparse_out = mock_layer.forward_impl(
                query_vllm, 
                torch.zeros(batch_spec.batch_size, kv_lora_rank, dtype=dtype, device=device),
                torch.zeros(batch_spec.batch_size, 1, qk_rope_head_dim, dtype=dtype, device=device),
                kv_cache, attn_metadata, sparse_out
            )
            outputs[backend_enum.name] = sparse_out

    # 7. Multi-way Assertions
    for name, out in outputs.items():
        if name == "DENSE": continue
        print(f"Comparing {name} vs DENSE...")
        torch.testing.assert_close(out, dense_output, rtol=1e-3, atol=1e-3)
        
    print(f"Success! Backends compared: {list(outputs.keys())}")

if __name__ == "__main__":
    test_triton_mla_sparse_parity("small_decode", "auto")
