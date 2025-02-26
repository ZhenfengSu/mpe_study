import json
from your_tokenizer import tokenizer  # 替换为实际使用的tokenizer

def filter_long_sequences(input_path, output_path, max_length=1024):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = []
    for sample in data:
        # 假设文本字段在"text"键下，根据实际数据结构调整
        tokens = tokenizer.encode(sample["text"], truncation=False)
        if len(tokens) <= max_length:
            filtered_data.append(sample)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

import json
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

def analyze_tokenized_lengths(json_path, tokenizer_path, save_path="length_distribution.png"):
    # 加载预训练的分词器
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # 读取JSON数据并处理
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lengths = []
    for i, item in enumerate(data):
        # 假设数据格式为{"text": "..."}
        text = item.get("text", "")
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens.ids))
        
        # 进度提示
        if i % 1000 == 0:
            print(f"Processing {i+1}/{len(data)} samples...")
    
    # 绘制统计图
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Tokenized Sequence Length')
    plt.ylabel('Sample Count')
    plt.title('Tokenized Text Length Distribution')
    plt.grid(axis='y', linestyle='--')
    
    # 重点标注max_length阈值
    plt.axvline(x=1024, color='r', linestyle='--', label='Max Length (1024)')
    plt.legend()
    
    # 保存并关闭
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved distribution plot to {save_path}")
