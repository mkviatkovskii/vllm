# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_and_maybe_dequant_weights, get_mla_dims)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.triton_sparse_mla import decode_sparse_attention_fwd

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

class TritonMLASparseBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if not use_sparse:
            return "TritonMLASparseBackend requires use_sparse=True"
        if device_capability.major < 8:
            return "TritonMLASparseBackend requires SM80+"
        
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            if not hasattr(vllm_config.model_config.hf_config, "index_topk"):
                return "TritonMLASparseBackend requires model with index_topk config"
        return None

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major >= 8

@dataclass
class TritonMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    topk_tokens: int
    req_id_per_token: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    topk_indices: torch.Tensor | None = None

class TritonMLASparseMetadataBuilder(AttentionMetadataBuilder[TritonMLASparseMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.block_size = kv_cache_spec.block_size
        
        # Buffer for request IDs per token
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonMLASparseMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        
        # Build req_id_per_token mapping efficiently
        q_start_loc = common_attn_metadata.query_start_loc
        diffs = torch.diff(q_start_loc)
        req_id_per_token = torch.repeat_interleave(
            torch.arange(len(diffs), dtype=torch.int32, device=self.device), 
            diffs
        )
            
        self.req_id_per_token_buffer[:num_tokens].copy_(
            req_id_per_token, non_blocking=True
        )
        
        return TritonMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            topk_tokens=self.topk_tokens,
            req_id_per_token=self.req_id_per_token_buffer[:num_tokens],
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
        )

class TritonMLASparseImpl(SparseMLAAttentionImpl[TritonMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        qk_head_dim: int = 192,
        v_head_dim: int = 128,
        kv_b_proj: "ColumnParallelLinear" = None,
        indexer: "Indexer | None" = None,
        q_pad_num_heads: int | None = None,
        **mla_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.kv_lora_rank = kv_lora_rank
        self.scale = float(scale)
        self.logit_cap = logits_soft_cap if logits_soft_cap is not None else 0.0
        self.kv_cache_dtype = kv_cache_dtype
        self.num_kv_heads = num_kv_heads
        
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj
        self.q_pad_num_heads = q_pad_num_heads
        
        assert indexer is not None
        self.topk_indices_buffer = indexer.topk_indices_buffer
        self.dcp_world_size = 1 # Default for V1 backend tests

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T

        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # Store for mock layer / tests
        self.W_UK = W_UK
        self.W_UV_raw = W_UV

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1)
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: TritonMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        
        if isinstance(q, tuple):
            # Already projected (ql_nope, q_pe)
            q_nope, q_pe = q
        else:
            # Need to project q with W_UK_T
            # q shape: (B, N, D)
            q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], 
                                   dim=-1)
            
            # Project q_nope: (B, N, P) @ (N, P, L) -> (B, N, L)
            # W_UK_T shape: (N, P, L)
            q_nope = q_nope.transpose(0, 1) # (N, B, P)
            q_nope = torch.bmm(q_nope, self.W_UK_T) # (N, B, L)
            q_nope = q_nope.transpose(0, 1) # (B, N, L)
            
        # Concatenate for Triton kernel: (B, N, L + P)
        q_attn = torch.cat([q_nope, q_pe], dim=-1)

        B = q_attn.shape[0]
        q_num_heads = q_attn.shape[1]
        
        # Get topk indices from indexer buffer
        topk_indices = self.topk_indices_buffer[:B]
        
        # Convert request-local topk indices to global KV cache slot IDs
        global_topk_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
            return_valid_counts=True,
        )

        # Intermediate attention output (B, N, L)
        attn_out = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q_attn.dtype, device=q_attn.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q_attn.dtype, device=q_attn.device)

        # Use 4 splits for better occupancy on large GPUs
        num_kv_splits = 1 if envs.VLLM_BATCH_INVARIANT else 4
        
        attn_logits = torch.empty(
            (B, q_num_heads, num_kv_splits, self.kv_lora_rank + 1),
            dtype=torch.float32,
            device=q_attn.device,
        )

        # Prepare cache tensors
        # Triton kernel expects (total_tokens, 1, head_size)
        kv_c_and_k_pe_cache_view = kv_c_and_k_pe_cache.view(-1, 1, kv_c_and_k_pe_cache.shape[-1])
        kv_c_cache = kv_c_and_k_pe_cache_view[..., : self.kv_lora_rank]

        decode_sparse_attention_fwd(
            q_attn,
            kv_c_and_k_pe_cache_view,
            kv_c_cache,
            global_topk_indices,
            attn_out,
            lse,
            valid_counts,
            attn_logits,
            num_kv_splits,
            self.scale,
            logit_cap=self.logit_cap,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
        )

        return attn_out, lse
