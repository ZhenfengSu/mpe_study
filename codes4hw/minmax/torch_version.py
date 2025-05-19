import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
# rearrange is used for tensor reshaping
from einops import rearrange
BLOCK = 256
# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->MiniMaxText01
class MiniMaxText01RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniMaxText01RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MiniMaxText01LightningAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        bias = False
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)

        self.out_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=bias)
        # silu
        self.act = nn.SiLU() 
        self.norm = MiniMaxText01RMSNorm(self.head_dim * self.num_heads)

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.head_dim * self.num_heads, bias=bias)
        self.output_gate = nn.Linear(self.hidden_size, self.head_dim * self.num_heads, bias=bias)

        # for inference only
        self.offset = 0
        self.layer_idx = layer_idx

    def forward(
            self,
            hidden_states,
            attn_mask: Optional[torch.Tensor] = None,  # (b, h, n, m)
            output_attentions: bool = False,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
            slope_rate: Optional[torch.Tensor] = None,
            **kwargs
    ):
        # if (not self.training) and (not do_eval):
        return self.inference(
            hidden_states,
            attn_mask,
            output_attentions,
            past_key_value,
            use_cache,
            slope_rate,
        )

    def inference(
            self,
            x,
            attn_mask: Optional[torch.Tensor] = None,  # (b, n)
            output_attentions: bool = False,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
            slope_rate: Optional[torch.Tensor] = None,  # (h, 1, 1)
    ):
        # x: b n d
        b, n, d = x.shape
        # linear map
        qkv = self.act(self.qkv_proj(x))
        new_shape = qkv.size()[:-1] + (self.num_heads, -1) # (b, n, h, d)
        qkv = qkv.view(*new_shape)
        q, k, v = torch.split(qkv, [self.head_dim] * 3, dim=3)
        q = q.transpose(1, 2) # 转置为 (b, h, n, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # for align with metaseq
        ratio = torch.exp(-slope_rate) # 进一步缩放

        # only use for the first time

        if past_key_value is None:
            slope_rate = slope_rate.to(torch.float32)
            if attn_mask is not None:
                v = v.masked_fill((1 - attn_mask).unsqueeze(1).unsqueeze(-1).to(torch.bool), 0)
            NUM_BLOCK = (n + BLOCK - 1) // BLOCK
            b, h, n, d = q.shape
            e = v.shape[-1] # e = 96
            # other
            array = torch.arange(BLOCK).to(q) + 1
            q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
            k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))
            index = array[:, None] - array[None, :]
            s_index = slope_rate * index[
                None,
                None,
            ]
            s_index = torch.where(index >= 0, -s_index, float("-inf"))
            diag_decay = torch.exp(s_index)

            kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device) # 初始化为全0
            output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device) # 初始化输出张量为空
            breakpoint()
            for i in range(NUM_BLOCK):
                si = i * BLOCK
                ei = min(si + BLOCK, n)
                m = ei - si
                qi = q[:, :, si:ei].contiguous()
                ki = k[:, :, si:ei].contiguous()
                vi = v[:, :, si:ei].contiguous() # 分块处理
                qkv_none_diag = torch.matmul(qi * q_decay[:, :m], kv).to(torch.float32)

                # diag
                qk = torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32) * diag_decay[:, :, :m, :m] # 矩阵乘法然后哈达玛积
                qkv_diag = torch.matmul(qk, vi.to(torch.float32))
                block_decay = torch.exp(-slope_rate * m)
                output[:, :, si:ei] = qkv_none_diag + qkv_diag
                kv = block_decay * kv + torch.matmul((ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi)

        else:
            kv = past_key_value
            output = []
            for i in range(n):
                kv = ratio * kv + torch.einsum(
                    "... n d, ... n e -> ... d e",
                    k[:, :, i:i + 1],
                    v[:, :, i:i + 1],
                )
                qkv = torch.einsum("... n e, ... e d -> ... n d", q[:, :, i:i + 1], kv.to(q.dtype))
                output.append(qkv)
            output = torch.concat(output, dim=-2)
        # reshape
        output = rearrange(output, "b h n d -> b n (h d)")
        # normalize
        output = self.norm(output)
        # gate
        output = F.sigmoid(self.output_gate(x)) * output
        # outproj
        output = self.out_proj(output)

        attn_weights = None

        return output, attn_weights, kv
    
class Config:
    def __init__(self, hidden_size, num_attention_heads, head_dim=None):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        
if __name__ == "__main__":
    hidden_size = 6144 
    num_attention_heads = 64
    head_dim = 128
    # Example configuration
    config = Config(hidden_size=hidden_size, num_attention_heads=num_attention_heads)

    attention_layer = MiniMaxText01LightningAttention(config)
    
    # Dummy input
    x = torch.randn(2, 100, hidden_size)  # (batch_size, sequence_length, hidden_size)
    slope_rate = torch.randn(num_attention_heads, 1, 1)  # (num_heads, 1, 1)
    output, attn_weights, kv = attention_layer(hidden_states = x, slope_rate=slope_rate)
    
    print("Output shape:", output.shape)
    print("KV shape:", kv.shape if kv is not None else "No KV returned")
