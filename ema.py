import torch
import torch.nn as nn

class PruneLayer(nn.Module):
    def __init__(self, layer, layer_type):
        super(PruneLayer, self).__init__()
        self.layer = layer
        self.layer_type = layer_type
        self.m = nn.Parameter(torch.tensor(1.0))
        self.m_old = self.m.clone().detach()  # 用于 EMA
        self.tau = 0.5  # 阈值
        self.beta = 10  # 控制 h 函数的陡峭程度

    def forward(self, x):
        h = self.get_h(self.m)
        out = h * self.layer(x) + (1 - h) * x
        return out

    def get_h(self, m):
        # 使用平滑的 Sigmoid 函数
        return torch.sigmoid(self.beta * (m - self.tau))

    def update_m_ema(self, alpha=0.9):
        with torch.no_grad():
            self.m.copy_(alpha * self.m_old + (1 - alpha) * self.m)
            self.m_old.copy_(self.m)

# 假设 optimizer 已定义

# 在每次优化器更新后，更新 m 的 EMA
for layer in model.modules():
    if isinstance(layer, PruneLayer):
        layer.update_m_ema(alpha=0.9)
