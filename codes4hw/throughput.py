import torch
from torch.utils.data import DataLoader, TensorDataset
import time

def measure_model_throughput(model, 
                             batch_size=64, 
                             image_size=(3, 224, 224),
                             num_iterations=100,
                             warmup_iterations=10,
                             verbose=True):
    """
    测量深度学习模型的吞吐率
    
    参数:
        model (torch.nn.Module): 待测试的PyTorch模型
        batch_size (int): 批次大小，默认256
        image_size (tuple): 输入图像尺寸，(通道, 高, 宽)，默认(3,224,224)
        num_iterations (int): 性能测试迭代次数，默认100
        warmup_iterations (int): 预热迭代次数，默认10
        verbose (bool): 是否显示详细输出，默认True
        
    返回:
        float: 吞吐率（每秒处理的图像数量）
    """
    # 设置设备
    device = next(model.parameters()).device
    model.eval()
    
    if verbose:
        print(f"开始测量模型吞吐率...")
        print(f"设备: {device}")
        print(f"批次大小: {batch_size}")
        print(f"输入尺寸: {image_size}")
        print(f"预热迭代: {warmup_iterations}, 测试迭代: {num_iterations}")
    
    # 检查GPU内存是否足够
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        # 保守估计：批次大小 * 每像素4字节 * 通道 * 宽 * 高 * 额外因子(8)
        required_memory = batch_size * 4 * image_size[0] * image_size[1] * image_size[2] * 8
        if required_memory > total_memory * 0.8:
            raise RuntimeError(f"GPU内存不足! 可用内存: {total_memory/1e9:.1f}GB, "
                            f"需求内存: {required_memory/1e9:.1f}GB。建议减小batch size")
    elif batch_size > 16 and verbose:
        print(f"警告: CPU上的大batch size ({batch_size})可能导致性能下降")
    
    # 创建虚拟输入
    batch_image_size = tuple([batch_size]) + image_size
    # images = torch.randn(256, 3, 224, 224)
    images = torch.randn(batch_image_size)
    print(images.shape)
    
    if verbose:
        print(f"生成虚拟输入")
        print("开始预热阶段...")
    images = images.to(device)
    # 预热阶段
    with torch.no_grad():
        for i in range(warmup_iterations):
            model(images)
    
    # 性能测试
    if verbose:
        print("开始性能测试...")
        
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_iterations):
            model(images)
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    end_time = time.time()
    
    # 计算结果
    elapsed_time = end_time - start_time
    total_samples = num_iterations * batch_size
    throughput = total_samples / elapsed_time
    
    if verbose:
        print("\n====== 测试结果 ======")
        print(f"完成迭代次数: {num_iterations}")
        print(f"总处理样本数: {total_samples}")
        print(f"总耗时: {elapsed_time:.4f}秒")
        print(f"吞吐率: {throughput:.2f} 图像/秒")
        print(f"吞吐率: {throughput/1000:.2f}k 图像/秒")
        print("="*30)
    
    return throughput

# 使用示例
if __name__ == "__main__":
    import torchvision
    
    # 示例1: 测试标准ViT模型
    vit_model = torchvision.models.vit_b_16(weights="DEFAULT")
    vit_model = vit_model.cuda().eval()  # 确保模型在GPU上
    
    print("测试标准ViT模型:")
    throughput = measure_model_throughput(vit_model)
    
    # 示例2: 测试ResNet模型
    resnet_model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    resnet_model = resnet_model.cuda().eval()
    
    print("\n测试ResNet-50模型:")
    throughput = measure_model_throughput(
        model=resnet_model,
        batch_size=128,
        image_size=(3, 224, 224),
        num_iterations=50
    )
    
    # 示例3: 测试自定义模型
    class CustomModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.fc = torch.nn.Linear(128*56*56, 1000)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    custom_model = CustomModel().cuda().eval()
    
    print("\n测试自定义模型:")
    throughput = measure_model_throughput(
        model=custom_model,
        batch_size=64,
        image_size=(3, 224, 224),
        num_iterations=100,
        verbose=True
    )
