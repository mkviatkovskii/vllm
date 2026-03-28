# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.mla.triton_mla_sparse import TritonMLASparseBackend
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend
from tests.v1.attention.test_mla_backends import (
    create_common_attn_metadata,
    create_and_prepopulate_kv_cache,
    create_vllm_config,
    run_attention_backend,
    BatchSpec,
    _convert_dtype_to_torch,
)
from vllm.config.vllm import set_current_vllm_config
from vllm.v1.kv_cache_interface import MLAAttentionSpec

class MockIndexer:
    def __init__(self, topk_indices_buffer: torch.Tensor):
        self.topk_indices_buffer = topk_indices_buffer

@pytest.mark.parametrize("batch_spec_name", ["small_decode", "medium_decode"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_triton_mla_sparse_vs_dense(
    batch_spec_name: str,
    kv_cache_dtype: str,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        pytest.skip("TRITON_MLA_SPARSE requires SM80+")

    # 1. Setup
    model = "deepseek-ai/DeepSeek-V2" # Use a model with index_topk
    hf_config_override = {
        "index_topk": 2048,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "num_attention_heads": 128,
    }
    
    BATCH_SPECS = {
        "small_decode": BatchSpec(seq_lens=[1024, 1024], query_lens=[1, 1]),
        "medium_decode": BatchSpec(seq_lens=[2048, 4096], query_lens=[1, 1]),
    }
    batch_spec = BATCH_SPECS[batch_spec_name]
    block_size = 16
    
    num_gpu_blocks = sum((s + block_size - 1) // block_size for s in batch_spec.seq_lens) + 10
    
    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        hf_config_override=hf_config_override,
    )
    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    
    num_q_heads = vllm_config.model_config.get_num_attention_heads()
    head_size = vllm_config.model_config.get_head_size()
    dtype = _convert_dtype_to_torch(vllm_config.model_config.dtype)
    kv_lora_rank = hf_config_override["kv_lora_rank"]
    qk_rope_head_dim = hf_config_override["qk_rope_head_dim"]
    qk_nope_head_dim = head_size - qk_rope_head_dim
    v_head_dim = qk_nope_head_dim # Typically same in MLA
    
    # 2. Generate data
    common_attn_metadata = create_common_attn_metadata(batch_spec, block_size, device)
    
    # Random queries
    query_vllm = torch.randn(
        batch_spec.batch_size, num_q_heads, qk_nope_head_dim + qk_rope_head_dim,
        dtype=dtype, device=device
    )
    
    # Context data for KV cache
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
        randomize_blocks=False, # For simplicity in index matching
        kv_cache_dtype=kv_cache_dtype,
    )

    # 3. Prepare Mock Indexer that selects ALL tokens (to match dense)
    # TritonMLASparseBackend needs an indexer in mla_args
    # We need to provide topk_indices that point to the full context
    max_seq_len = max(batch_spec.seq_lens)
    topk_tokens = vllm_config.model_config.hf_config.index_topk
    
    topk_indices_buffer = torch.full(
        (batch_spec.batch_size, topk_tokens), -1, dtype=torch.int32, device=device
    )
    
    for i, s_len in enumerate(batch_spec.seq_lens):
        # Local indices for this sequence: 0, 1, ..., s_len-1
        # (Actually, in sparse MLA, it might only attend to context tokens, 
        # but TritonMLASparseImpl supports up to s_len-1)
        indices = torch.arange(s_len, dtype=torch.int32, device=device)
        topk_indices_buffer[i, :s_len] = indices

    mock_indexer = MockIndexer(topk_indices_buffer)

    # 4. Run Dense Backend
    # Need to mock kv_b_proj
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_q_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
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
        torch.empty(0), # kv_c_vllm - not used in decode
        torch.empty(0), # k_pe_vllm - not used in decode
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

    # 5. Run Sparse Backend
    # We need to hack run_attention_backend to pass indexer to TritonMLASparseImpl
    # Or manually call it.
    
    from vllm.v1.attention.backends.mla.triton_mla_sparse import TritonMLASparseImpl, TritonMLASparseMetadataBuilder
    
    with set_current_vllm_config(vllm_config):
        impl = TritonMLASparseImpl(
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
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_b_proj=mock_kv_b_proj,
            indexer=mock_indexer,
        )
        impl.process_weights_after_loading(dtype)
        if impl.dcp_world_size == -1:
            impl.dcp_world_size = 1
            
        from tests.v1.attention.test_mla_backends import MockSparseMLAAttentionLayer
        mock_layer = MockSparseMLAAttentionLayer(
            impl=impl,
            num_heads=num_q_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            device=device,
            W_UK=impl.W_UK,
            W_UV=impl.W_UV,
            q_scale=1.0,
            k_scale=1.0,
        )
        
        builder = TritonMLASparseMetadataBuilder(kv_cache_spec, ["layer1"], vllm_config, device)
        attn_metadata = builder.build(0, common_attn_metadata)
        
        sparse_output = torch.empty_like(dense_output)
        sparse_output = mock_layer.forward_impl(
            query_vllm, torch.empty(0, device=device), torch.empty(0, device=device), 
            kv_cache, attn_metadata, sparse_output
        )

    # 6. Compare
    print(f"Dense output sum: {dense_output.sum().item()}")
    print(f"Sparse output sum: {sparse_output.sum().item()}")
    
    # Sparse vs Dense comparison
    # They should be very close if sparse selects all tokens
    rtol = 1e-3
    atol = 1e-3
    torch.testing.assert_close(sparse_output, dense_output, rtol=rtol, atol=atol)
    print("Success: Sparse matches Dense!")

if __name__ == "__main__":
    test_triton_mla_sparse_vs_dense("small_decode", "auto")
