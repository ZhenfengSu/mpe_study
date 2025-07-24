import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel文件
file_path = '计算核心变化.xlsx'  # 请替换为您的文件路径

# 读取两个sheet
sm_data = pd.read_excel(file_path, sheet_name='SM')
ai_core_data = pd.read_excel(file_path, sheet_name='AI_CORE')

# 获取数据
sm_versions = sm_data.iloc[:, 0]  # 第一列：芯片版本
sm_numbers = sm_data.iloc[:, 1]   # 第二列：SM数量

ai_versions = ai_core_data.iloc[:, 0]  # 第一列：芯片版本
ai_numbers = ai_core_data.iloc[:, 1]   # 第二列：AI Core数量

# 设置配色方案
color_blue = '#367EB8'    # 蓝色 - SM数量
color_red = '#C00000'     # 红色 - AI Core数量
text_color = '#000000'    # 文字颜色
grid_color = '#E0E0E0'    # 网格颜色

# 创建包含两个子图的图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# ========== 第一个子图：NVIDIA GPU ==========
ax1.plot(range(len(sm_versions)), sm_numbers, 
         color=color_blue, marker='o', linewidth=3, 
         markersize=8, label='SM Numbers')

# 在SM数据点上添加数值标签
for i, (x, y) in enumerate(zip(range(len(sm_versions)), sm_numbers)):
    ax1.annotate(f'{y}', 
                xy=(x, y), 
                xytext=(0, 12),  # 向上12像素
                textcoords='offset points',
                fontsize=10,
                color=color_blue,
                fontweight='bold',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color_blue, alpha=0.8))

# 设置第一个子图的标签和格式
ax1.set_xlabel('GPU Version', fontsize=14, color=text_color, fontweight='bold')
ax1.set_ylabel('SM Numbers', fontsize=14, color=text_color, fontweight='bold')
ax1.set_title('NVIDIA GPU', fontsize=16, fontweight='bold', color=text_color, pad=25)

# 设置X轴标签
ax1.set_xticks(range(len(sm_versions)))
ax1.set_xticklabels(sm_versions, rotation=45, ha='right', fontsize=11)
ax1.tick_params(axis='y', labelcolor=text_color, labelsize=11)
ax1.tick_params(axis='x', labelcolor=text_color)

# 设置网格
ax1.grid(True, color=grid_color, linestyle='-', alpha=0.4)
ax1.set_axisbelow(True)

# ========== 第二个子图：Huawei NPU ==========
ax2.plot(range(len(ai_versions)), ai_numbers, 
         color=color_red, marker='s', linewidth=3, 
         markersize=8, label='AI Core Numbers')

# 在AI Core数据点上添加数值标签
for i, (x, y) in enumerate(zip(range(len(ai_versions)), ai_numbers)):
    ax2.annotate(f'{y}', 
                xy=(x, y), 
                xytext=(0, 12),  # 向上12像素
                textcoords='offset points',
                fontsize=10,
                color=color_red,
                fontweight='bold',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color_red, alpha=0.8))

# 设置第二个子图的标签和格式
ax2.set_xlabel('NPU Version', fontsize=14, color=text_color, fontweight='bold')
ax2.set_ylabel('AI Core Numbers', fontsize=14, color=text_color, fontweight='bold')
ax2.set_title('Huawei NPU', fontsize=16, fontweight='bold', color=text_color, pad=25)

# 设置X轴标签
ax2.set_xticks(range(len(ai_versions)))
ax2.set_xticklabels(ai_versions, rotation=45, ha='right', fontsize=11)
ax2.tick_params(axis='y', labelcolor=text_color, labelsize=11)
ax2.tick_params(axis='x', labelcolor=text_color)

# 设置网格
ax2.grid(True, color=grid_color, linestyle='-', alpha=0.4)
ax2.set_axisbelow(True)

# 设置整体标题
fig.suptitle('Evolution of Computing Cores Across Chip Versions', 
             fontsize=18, fontweight='bold', color=text_color, y=0.98)

# 设置背景颜色
fig.patch.set_facecolor('#FEFEFE')
ax1.set_facecolor('#FEFEFE')
ax2.set_facecolor('#FEFEFE')

# 调整子图之间的间距和布局
plt.tight_layout()
plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.3)  # 增加顶部空间，调整间距

# 显示图表
plt.show()

# 保存图表
plt.savefig('computing_cores_evolution_subplots.png', dpi=300, bbox_inches='tight', 
            facecolor='#FEFEFE', edgecolor='none')

print("图表绘制完成！")
print(f"NVIDIA GPU SM数据点数量: {len(sm_numbers)}")
print(f"Huawei NPU AI Core数据点数量: {len(ai_numbers)}")
