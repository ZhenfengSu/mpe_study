import torch

def standardize_tensor(tensor, eps=1e-8):
    """
    对给定的 tensor 进行标准化，使其均值为 0，标准差为 1。
    
    参数：
        tensor (torch.Tensor): 要标准化的输入张量。
        eps (float): 为了防止分母为零，添加的小常数。
        
    返回：
        torch.Tensor: 标准化后的张量。
    """
    # 计算均值和标准差
    mean = tensor.mean()
    std = tensor.std()

    # 如果标准差为 0，设置为一个很小的值 eps
    std = std if std > eps else eps
    
    # 标准化
    standardized_tensor = (tensor - mean) / std
    
    return standardized_tensor
