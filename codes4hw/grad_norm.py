import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class PruneLayer(nn.Module):
    def __init__(self, layer, layer_type):
        super(PruneLayer, self).__init__()
        self.layer = layer  # 原始层，例如 attention 或 activation 层
        self.layer_type = layer_type  # 层类型，"attention" 或 "activation"
        self.m = nn.Parameter(torch.tensor(1.0))  # 可训练的权重参数 m_i

    def forward(self, x):
        h = self.get_h(self.m)
        out = h * self.layer(x) + (1 - h) * x
        return out

    def get_h(self, m):
        # 这里可以根据需要定义您的 h(m) 函数
        # 为了简化，假设 h(m) = sigmoid(m)
        return torch.sigmoid(m)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        # 假设有一些 attention 层和 activation 层
        for i in range(12):
            if i % 2 == 0:
                # attention 层
                layer = nn.Linear(768, 768)  # 示例
                self.layers.append(PruneLayer(layer, "attention"))
            else:
                # activation 层
                layer = nn.ReLU()
                self.layers.append(PruneLayer(layer, "activation"))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def gradient_normalization(model):
    # 收集不同层类型的 m_i 的梯度
    attention_grads = []
    activation_grads = []

    for layer in model.modules():
        if isinstance(layer, PruneLayer):
            if layer.layer_type == "attention":
                attention_grads.append(layer.m.grad)
            elif layer.layer_type == "activation":
                activation_grads.append(layer.m.grad)

    # 将梯度转换为张量
    if attention_grads:
        attention_grads_tensor = torch.stack(attention_grads)
    else:
        attention_grads_tensor = None

    if activation_grads:
        activation_grads_tensor = torch.stack(activation_grads)
    else:
        activation_grads_tensor = None

    # 在 DDP 模式下，需要同步所有进程的梯度信息
    if dist.is_initialized():
        if attention_grads_tensor is not None:
            dist.all_reduce(attention_grads_tensor, op=dist.ReduceOp.SUM)
            attention_grads_tensor /= dist.get_world_size()

        if activation_grads_tensor is not None:
            dist.all_reduce(activation_grads_tensor, op=dist.ReduceOp.SUM)
            activation_grads_tensor /= dist.get_world_size()

    # 分别计算每个组的梯度范数（或均值、标准差等）
    if attention_grads_tensor is not None:
        attention_norm = attention_grads_tensor.norm()
        # 避免除以零
        attention_norm = attention_norm if attention_norm > 0 else 1.0
    else:
        attention_norm = 1.0

    if activation_grads_tensor is not None:
        activation_norm = activation_grads_tensor.norm()
        activation_norm = activation_norm if activation_norm > 0 else 1.0
    else:
        activation_norm = 1.0

    # 对每个 m_i 的梯度进行归一化
    for layer in model.modules():
        if isinstance(layer, PruneLayer):
            if layer.m.grad is not None:
                if layer.layer_type == "attention":
                    layer.m.grad /= attention_norm
                elif layer.layer_type == "activation":
                    layer.m.grad /= activation_norm

def main():
    # 初始化 DDP
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    model = Model().cuda()
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 示例输入和目标
    x = torch.randn(32, 768).cuda()
    target = torch.randn(32, 768).cuda()

    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, target)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 进行梯度归一化
    gradient_normalization(model)

    # 优化器更新参数
    optimizer.step()

if __name__ == '__main__':
    main()
