import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union
BLOCK = 256
from einops import rearrange
from light_attn_mine import inference
class linear_attn_custom(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                qkv,
                attn_mask: Optional[torch.Tensor] = None,  # (b, n)
                slope_rate: Optional[torch.Tensor] = None,  # (h, 1, 1)
                num_heads: Optional[int] = 64,
                head_dim: Optional[int] = 96,):
        new_shape = qkv.size()[:-1] + (num_heads, -1) # (b, n, h, d)
        qkv = qkv.view(*new_shape)
        q, k, v = torch.split(qkv, [head_dim] * 3, dim=3)
        q = q.transpose(1, 2) # 转置为 (b, h, n, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # only use for the first time
        '''---------------------------------------------------linear attn----------------------------------------------------------------'''
        # if past_key_value is None:
        b, h, n, d = q.shape
        slope_rate = slope_rate.to(torch.float32)
        if attn_mask is not None:
            v = v.masked_fill((1 - attn_mask).unsqueeze(1).unsqueeze(-1).to(torch.bool), 0)
        NUM_BLOCK = (n + BLOCK - 1) // BLOCK
        
        e = v.shape[-1] # e = 96
        # other
        array = torch.arange(BLOCK).to(q) + 1 # [1, 2, ..., BLOCK]
        q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
        k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))
        index = array[:, None] - array[None, :]
        s_index = slope_rate * index[
            None,
            None,
        ]
        s_index = torch.where(index >= 0, -s_index, float("-inf"))
        diag_decay = torch.exp(s_index)
        kv_state = []
        kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device) # 初始化为全0
        output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device) # 初始化输出张量为空
        # output  = []
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
            # output.append(qkv_none_diag + qkv_diag)
            # update kv
            kv = block_decay * kv + torch.matmul((ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi)
            kv_state.append(kv)
        # output = torch.concat(output, dim=-2)
        '''---------------------------------------------------linear attn----------------------------------------------------------------'''
        # kv_state转化为tensor
        kv_state = torch.stack(kv_state, dim=0)
        ctx.save_for_backward(q, k, v, slope_rate, kv_state)
        # reshape
        output = rearrange(output, "b h n d -> b n (h d)")
        # ctx.save_for_backward(result)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        # result, = ctx.saved_tensors
        q, k, v, slope_rate, kv_state = ctx.saved_tensors
        b, h, n, d = q.shape
        
        # reshape grad_output
        new_shape = grad_output.size()[:-1] + (h, -1) # (b, n, h, d)
        grad_output = grad_output.view(*new_shape)
        grad_output = grad_output.transpose(1, 2) # 转置为 (b, h, n, d)
        
        e = v.shape[-1]
        NUM_BLOCK = (n + BLOCK - 1) // BLOCK
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
        d_kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device) # 初始化为全0
        d_Q = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device) # 初始化输出张量为空
        d_K = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
        d_V = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
        for i in range(NUM_BLOCK, 0, -1):
            ei = min(i * BLOCK, n)
            si = max(0, ei - BLOCK)
            m = ei - si
            qi = q[:, :, si:ei].contiguous()
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()
            do_i = grad_output[:, :, si:ei].contiguous()
            kv_i = kv_state[i - 1]
            # q intra
            ov_mask = torch.matmul(do_i, vi.transpose(-1, -2)).to(torch.float32) *  diag_decay[:, :, :m, :m]
            q_non_diag = torch.matmul(ov_mask, ki).to(torch.float32)
            # q inter
            # q_diag = torch.matmul(do_i*q_decay[:, :m], kv_i.transpose(-1, -2)).to(torch.float32)
            q_diag = torch.matmul(do_i*q_decay[:, :m], kv_i.transpose(-1, -2)).to(torch.float32)
            d_Q[:, :, si:ei] = q_non_diag + q_diag
            # k intra
            k_non_diag = torch.matmul(ov_mask.transpose(-1,-2), qi).to(torch.float32)
            # k inter
            k_diag = torch.matmul(vi*k_decay[:, -m:], d_kv.transpose(-1, -2)).to(torch.float32)
            d_K[:, :, si:ei] = k_non_diag + k_diag
            # v intra
            qk_mask = torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32) * diag_decay[:, :, :m, :m]
            v_non_diag = torch.matmul(qk_mask.transpose(-1,-2), do_i).to(torch.float32)
            # v inter
            v_diag = torch.matmul(ki*k_decay[:, -m:], d_kv).to(torch.float32)
            # d_V[:, :, si:ei] = v_non_diag + v_diag
            d_V[:, :, si:ei] = v_non_diag + v_diag
            # update dkv
            block_decay = torch.exp(-slope_rate * m)
            d_kv = block_decay * d_kv + torch.matmul((qi * q_decay[:, :m]).transpose(-1, -2).to(vi.dtype), do_i)
        # dq dk dv 拼接
        d_qkv = torch.cat([d_Q, d_K, d_V], dim=-1)
        # reshape
        d_qkv = rearrange(d_qkv, "b h n d -> b n (h d)")
        return d_qkv, None, None, None, None
def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closet_power_of_2 = 2 ** math.floor(math.log2(n))
        return (get_slopes_power_of_2(closet_power_of_2) + get_slopes(2*closet_power_of_2)[0::2][:n - closet_power_of_2])
if __name__ == "__main__":
    # 测试代码
    hidden_size = 6144 
    num_attention_heads = 64
    head_dim = 96
    batch_size = 2
    seq_len = 100
    x = torch.randn(2, 100, hidden_size*3, requires_grad=True).to(torch.float32)  # (batch_size, sequence_length, hidden_size)
    # 复制x
    x_clone = x.clone().detach().requires_grad_(True)
    # slope_rate = torch.randn(num_attention_heads, 1, 1, requires_grad=False).to(torch.float32)   # (num_heads, 1, 1)
    slope_rate =torch.tensor(get_slopes(num_attention_heads),dtype= torch.float32).reshape(num_attention_heads, 1, 1).to(x) # (num_heads, 1, 1)
    output = linear_attn_custom.apply(x, None, slope_rate, num_attention_heads, head_dim)
    print(output.shape)  # 输出形状应该是 (b, n, h * d)
    # 反向传播测试
    output.sum().backward()
    print(x.grad)  # 检查梯度是否计算正确
    # 打印shape
    print("x grad shape:", x.grad.shape)  # 应该是 (b, n, h * d)
    
    # 用inference函数测试
    output_clone = inference(qkv = x_clone,attn_mask=None, slope_rate=slope_rate)
    print("Output shape:", output.shape)
    # 反向传播测试
    output_clone.sum().backward()
    print(x_clone.grad)  # 检查梯度是否计算正确
    # 打印shape
    print("x_clone grad shape:", x_clone.grad.shape)  # 应该是 (b, n, h * d)
    # 比较两个输出
    print("Output difference:", torch.allclose(output, output_clone, atol=1e-6))  # 检查两个输出是否相近
    # 比较两个梯度
    print("Gradient difference:", torch.allclose(x.grad, x_clone.grad, atol=1e-6))  # 检查两个梯度是否相近
    q_grad, k_grad, v_grad = torch.split(x.grad, [hidden_size] * 3, dim=-1)
    q_grad_clone, k_grad_clone, v_grad_clone = torch.split(x_clone.grad, [hidden_size] * 3, dim=-1)
    bool_q = q_grad == q_grad_clone
    print(bool_q)
    bool_k = k_grad == k_grad_clone
    print(bool_k)
    bool_v = v_grad == v_grad_clone
    print(bool_v)
    
    # 检查bool v不为true的位置
    bool_v_false = torch.where(bool_v == False)
    print("bool_v_false:", bool_v_false)
    false_num = bool_v_false[0].shape[0]
    print("false_num:", false_num)
    all_num = v_grad.shape[0] * v_grad.shape[1] * v_grad.shape[2]
    print("all_num:", all_num)
    # rate
    rate = false_num / all_num
    print("rate:", rate)
