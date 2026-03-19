"""
可视化工具

实现训练过程的可视化：
1. 损失曲线
2. 准确率对比
3. 注意力权重热力图
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Optional
from pathlib import Path


# 设置中文字体（如果可用）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Path] = None,
    title: str = "Training and Validation Loss"
):
    """
    绘制训练和验证损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径（如果为 None，则显示图像）
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失曲线已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_accuracy_comparison(
    accuracy_data: Dict[str, List[float]],
    labels: List[str],
    save_path: Optional[Path] = None,
    title: str = "Accuracy Comparison",
    ylabel: str = "Accuracy (%)"
):
    """
    绘制准确率对比柱状图

    Args:
        accuracy_data: 准确率数据字典
            {
                '3+3': [95.2, 96.1, 97.3],  # 3个模型的准确率
                '3+4': [92.1, 93.5, 94.2],
                ...
            }
        labels: 柱状图的标签（如模型名称）
        save_path: 保存路径
        title: 图表标题
        ylabel: Y轴标签
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(accuracy_data))
    width = 0.8 / len(labels)

    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    for i, label in enumerate(labels):
        values = [accuracy_data[key][i] for key in accuracy_data.keys()]
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=colors[i])

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Addition Type', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(list(accuracy_data.keys()), fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"准确率对比图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_map(
    attention_weights: torch.Tensor,
    tokens_src: List[str],
    tokens_tgt: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """
    绘制注意力权重热力图

    Args:
        attention_weights: 注意力权重 [num_heads, tgt_len, src_len]
        tokens_src: 源序列的 token 列表
        tokens_tgt: 目标序列的 token 列表
        layer_idx: 层索引（用于标题）
        head_idx: 头索引（选择要可视化的头）
        save_path: 保存路径
        title: 图表标题
    """
    # 选择指定头的注意力权重
    attn = attention_weights[head_idx].cpu().numpy()

    # 创建图表
    fig, ax = plt.subplots(figsize=(max(len(tokens_src), len(tokens_tgt)), max(len(tokens_tgt), len(tokens_src)) / 2))

    # 绘制热力图
    sns.heatmap(attn, xticklabels=tokens_src, yticklabels=tokens_tgt,
                cmap='viridis', cbar_kws={'label': 'Attention Weight'},
                linewidths=0.5, linecolor='gray', ax=ax)

    if title is None:
        title = f'Attention Map (Layer {layer_idx}, Head {head_idx})'
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Source Tokens', fontsize=12)
    ax.set_ylabel('Target Tokens', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力热力图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_multi_head_attention(
    attention_weights: torch.Tensor,
    tokens_src: List[str],
    tokens_tgt: List[str],
    layer_idx: int = 0,
    save_path: Optional[Path] = None
):
    """
    绘制多头注意力的热力图（所有头）

    Args:
        attention_weights: 注意力权重 [num_heads, tgt_len, src_len]
        tokens_src: 源序列的 token 列表
        tokens_tgt: 目标序列的 token 列表
        layer_idx: 层索引
        save_path: 保存路径
    """
    num_heads = attention_weights.size(0)

    # 计算子图布局
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(num_heads):
        ax = axes[i]
        attn = attention_weights[i].cpu().numpy()

        sns.heatmap(attn, xticklabels=tokens_src, yticklabels=tokens_tgt,
                   cmap='viridis', cbar=True, linewidths=0.5,
                   linecolor='gray', ax=ax)
        ax.set_title(f'Head {i}', fontsize=10)
        ax.set_xlabel('Source', fontsize=9)
        ax.set_ylabel('Target', fontsize=9)

    # 隐藏多余的子图
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Multi-Head Attention (Layer {layer_idx})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多头注意力热力图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


# 测试
if __name__ == '__main__':
    print("测试可视化工具")

    # 测试损失曲线
    print("\n" + "="*60)
    print("测试损失曲线")
    print("="*60)

    train_losses = [2.5, 2.0, 1.6, 1.3, 1.1, 0.9, 0.8, 0.7, 0.6, 0.5]
    val_losses = [2.7, 2.2, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]

    plot_loss_curves(
        train_losses,
        val_losses,
        save_path=Path('outputs/figures/test_loss_curves.png'),
        title="Test Loss Curves"
    )

    # 测试准确率对比
    print("\n" + "="*60)
    print("测试准确率对比")
    print("="*60)

    accuracy_data = {
        '3+3': [95.2, 96.1, 97.3],
        '3+4': [92.1, 93.5, 94.2],
        '4+3': [91.8, 92.9, 93.8],
        '4+4': [89.5, 91.2, 92.5],
    }
    labels = ['Transformer', 'GPT-Small', 'GPT-Medium']

    plot_accuracy_comparison(
        accuracy_data,
        labels,
        save_path=Path('outputs/figures/test_accuracy_comparison.png'),
        title="Test Accuracy Comparison",
        ylabel="Accuracy (%)"
    )

    # 测试注意力热力图
    print("\n" + "="*60)
    print("测试注意力热力图")
    print("="*60)

    # 创建模拟的注意力权重
    num_heads = 4
    tgt_len = 5
    src_len = 7

    attention_weights = torch.rand(num_heads, tgt_len, src_len)
    # 归一化
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

    tokens_src = list("123+456=")
    tokens_tgt = list("579")

    plot_attention_map(
        attention_weights,
        tokens_src,
        tokens_tgt,
        layer_idx=0,
        head_idx=0,
        save_path=Path('outputs/figures/test_attention_map.png')
    )

    plot_multi_head_attention(
        attention_weights,
        tokens_src,
        tokens_tgt,
        layer_idx=0,
        save_path=Path('outputs/figures/test_multi_head_attention.png')
    )

    print("\n所有测试图表已保存到 outputs/figures/")
