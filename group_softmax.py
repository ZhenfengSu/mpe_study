import torch  
import numpy as np  

def apply_softmax_to_odd_and_even_positions(inputs):  
    # 转换为 PyTorch 张量以使用 softmax  
    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)  

    # 提取偶数和奇数索引位置的元素  
    even_indices = tensor_inputs[::2]  # 偶数位置的值  
    odd_indices = tensor_inputs[1::2]  # 奇数位置的值  

    # 应用 softmax  
    softmax_even = torch.softmax(even_indices, dim=0)  
    softmax_odd = torch.softmax(odd_indices, dim=0)  

    # 创建结果列表  
    result = np.empty_like(inputs, dtype=np.float32)  

    # 将 softmax 的结果放回原列表的相应位置  
    result[::2] = softmax_even.numpy()  # 将偶数位置替换  
    result[1::2] = softmax_odd.numpy()  # 将奇数位置替换  

    return result  

# 示例用法  
input_list = [1.0, 3.0, 2.0, 4.0, 1.5, 6.0]  
output_list = apply_softmax_to_odd_and_even_positions(input_list)  
print("Original list:", input_list)  
print("Modified list:", output_list)
