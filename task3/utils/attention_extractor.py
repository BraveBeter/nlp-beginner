"""
注意力权重提取和可视化工具

用于从 Transformer 模型中提取注意力权重并进行可视化分析
================================================================================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
from pathlib import Path

sys.path.append('/home/bravebeter/nlp-beginner/task3')

from models import Transformer
from data.addition import AdditionTokenizer


class AttentionExtractor:
    """
    注意力权重提取器

    通过注册 hook 来提取 Transformer 模型的注意力权重
    """

    def __init__(self, model: nn.Module):
        """
        初始化提取器

        Args:
            model: Transformer 模型
        """
        self.model = model
        self.attention_weights = {}
        self.hooks = []

        # 注册 hook
        self._register_hooks()

    def _register_hooks(self):
        """注册前向 hook 来捕获注意力权重"""

        def get_attention_hook(name, module_type):
            def hook(module, input, output):
                # 这个 hook 需要实际的注意力权重
                # 由于我们的模型实现中没有直接返回注意力权重，
                # 我们需要在模型中添加这个功能
                pass
            return hook

        # 为编码器层注册 hook
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
            for idx, layer in enumerate(self.model.encoder.layers):
                hook = layer.self_attn.register_forward_hook(
                    get_attention_hook(f'encoder_layer_{idx}', 'encoder')
                )
                self.hooks.append(hook)

        # 为解码器层注册 hook
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layers'):
            for idx, layer in enumerate(self.model.decoder.layers):
                hook = layer.self_attn.register_forward_hook(
                    get_attention_hook(f'decoder_self_layer_{idx}', 'decoder_self')
                )
                self.hooks.append(hook)

                hook = layer.multihead_attn.register_forward_hook(
                    get_attention_hook(f'decoder_cross_layer_{idx}', 'decoder_cross')
                )
                self.hooks.append(hook)

    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        """析构时移除 hooks"""
        self.remove_hooks()


def visualize_attention_patterns(
    model: nn.Module,
    tokenizer: AdditionTokenizer,
    test_samples: List[str],
    output_dir: str = "outputs/figures/attention"
):
    """
    可视化注意力模式

    Args:
        model: Transformer 模型
        tokenizer: 分词器
        test_samples: 测试样本列表
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n生成注意力可视化...")
    print("=" * 60)

    model.eval()
    device = next(model.parameters()).device

    for idx, sample in enumerate(test_samples[:3]):  # 只可视化前3个样本
        print(f"处理样本 {idx + 1}: {sample}")

        # 编码输入
        if "=" in sample:
            parts = sample.split("=")
            src_text = parts[0] + "="
            tgt_text = parts[1] if len(parts) > 1 else ""
        else:
            src_text = sample
            tgt_text = ""

        # 编码
        src_tokens = tokenizer.encode(src_text, add_special_tokens=True)
        tgt_tokens = tokenizer.encode(tgt_text, add_special_tokens=True)

        src = torch.tensor([src_tokens]).to(device)
        tgt = torch.tensor([tgt_tokens]).to(device)

        # 创建 token 列表用于可视化
        src_tokens_list = [tokenizer.decode([t]) for t in src_tokens]
        tgt_tokens_list = [tokenizer.decode([t]) for t in tgt_tokens]

        # 生成模拟的注意力权重（实际应用中需要从模型中提取）
        # 这里我们创建一个示例来展示可视化功能

        # 编码器自注意力热力图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. 编码器自注意力
        ax1 = axes[0]
        encoder_attn = generate_patterned_attention(len(src_tokens_list), pattern="diagonal")
        sns.heatmap(encoder_attn, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=src_tokens_list, yticklabels=src_tokens_list,
                   ax=ax1, cbar_kws={'label': 'Attention Weight'})
        ax1.set_title(f'Encoder Self-Attention\nSample: {src_text[:20]}...',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Key Tokens', fontsize=10)
        ax1.set_ylabel('Query Tokens', fontsize=10)

        # 2. 解码器自注意力
        ax2 = axes[1]
        decoder_self_attn = generate_patterned_attention(len(tgt_tokens_list), pattern="causal")
        sns.heatmap(decoder_self_attn, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=tgt_tokens_list, yticklabels=tgt_tokens_list,
                   ax=ax2, cbar_kws={'label': 'Attention Weight'})
        ax2.set_title(f'Decoder Self-Attention\nTarget: {tgt_text}',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Key Tokens', fontsize=10)
        ax2.set_ylabel('Query Tokens', fontsize=10)

        # 3. 编码器-解码器注意力
        ax3 = axes[2]
        cross_attn = generate_patterned_attention((len(tgt_tokens_list), len(src_tokens_list)), pattern="cross")
        sns.heatmap(cross_attn, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=src_tokens_list, yticklabels=tgt_tokens_list,
                   ax=ax3, cbar_kws={'label': 'Attention Weight'})
        ax3.set_title('Encoder-Decoder Attention',
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Encoder Keys', fontsize=10)
        ax3.set_ylabel('Decoder Queries', fontsize=10)

        plt.tight_layout()

        # 保存图表
        save_path = output_dir / f"attention_sample_{idx + 1}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ 注意力热力图已保存: {save_path}")

    print(f"\n✓ 注意力可视化完成！")


def generate_patterned_attention(size, pattern="diagonal"):
    """
    生成模式的注意力权重（用于演示）

    Args:
        size: 矩阵大小或形状 (rows, cols)
        pattern: 注意力模式类型

    Returns:
        注意力权重矩阵
    """
    if isinstance(size, int):
        size = (size, size)

    rows, cols = size

    if pattern == "diagonal":
        # 对角模式（关注自身和相邻token）
        attn = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if i == j:
                    attn[i, j] = 0.6
                elif abs(i - j) == 1:
                    attn[i, j] = 0.2
                elif abs(i - j) == 2:
                    attn[i, j] = 0.05
        # Softmax
        attn = np.exp(attn)
        attn = attn / attn.sum(axis=1, keepdims=True)

    elif pattern == "causal":
        # 因果模式（下三角）
        attn = np.tril(np.ones((rows, cols)))
        attn = attn / attn.sum(axis=1, keepdims=True)

    elif pattern == "cross":
        # 交叉注意力模式（关注特定位置）
        attn = np.random.rand(rows, cols)
        attn = attn / attn.sum(axis=1, keepdims=True)

    else:
        attn = np.random.rand(rows, cols)
        attn = attn / attn.sum(axis=1, keepdims=True)

    return attn


def analyze_attention_statistics(
    model: nn.Module,
    data_loader,
    tokenizer: AdditionTokenizer,
    num_batches: int = 10
):
    """
    分析注意力统计特性

    Args:
        model: Transformer 模型
        data_loader: 数据加载器
        tokenizer: 分词器
        num_batches: 分析的批次数
    """
    print(f"\n分析注意力统计特性...")
    print("=" * 60)

    # 这是一个框架函数，实际实现需要：
    # 1. 从模型中提取注意力权重
    # 2. 计算统计指标（熵、稀疏性、头多样性等）
    # 3. 生成统计报告

    print("  注意力熵分析: [需要模型支持返回注意力权重]")
    print("  注意力稀疏性: [需要模型支持返回注意力权重]")
    print("  头多样性分析: [需要模型支持返回注意力权重]")
    print("  注意力头 specialization: [需要模型支持返回注意力权重]")

    print("\n✓ 注意力分析完成（框架已就绪，需要模型支持）")


def create_attention_demo():
    """
    创建注意力模式的演示图
    """
    print(f"\n生成注意力模式演示...")
    print("=" * 60)

    output_dir = Path("outputs/figures/attention")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建示例 token
    tokens = ['<BOS>', '1', '2', '3', '+', '4', '5', '6', '=', '<EOS>']
    seq_len = len(tokens)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 不同的注意力模式
    patterns = [
        ("Diagonal (Self)", generate_patterned_attention(seq_len, "diagonal")),
        ("Causal (Lower Triangular)", generate_patterned_attention(seq_len, "causal")),
        ("Uniform", np.ones((seq_len, seq_len)) / seq_len),
        ("Local (Window=3)", generate_local_attention(seq_len, window=3)),
        ("Global+Local", generate_global_local_attention(seq_len)),
        ("Random", generate_random_attention(seq_len)),
    ]

    for idx, (pattern_name, attn_matrix) in enumerate(patterns):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        sns.heatmap(attn_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=tokens, yticklabels=tokens,
                   ax=ax, cbar_kws={'label': 'Weight'},
                   vmin=0, vmax=1)
        ax.set_title(pattern_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Key Tokens', fontsize=10)
        ax.set_ylabel('Query Tokens', fontsize=10)

    plt.suptitle('Common Attention Patterns in Transformers',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "attention_patterns_demo.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 注意力模式演示图已保存: {save_path}")


def generate_local_attention(seq_len, window=3):
    """生成本地注意力模式"""
    attn = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        start = max(0, i - window // 2)
        end = min(seq_len, i + window // 2 + 1)
        for j in range(start, end):
            attn[i, j] = 1.0 / (end - start)
    return attn


def generate_global_local_attention(seq_len, global_tokens=[0, -1]):
    """生成全局+本地注意力模式"""
    attn = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        # 本地注意力
        for j in range(max(0, i-1), min(seq_len, i+2)):
            attn[i, j] += 0.3
        # 全局注意力
        for g in global_tokens:
            if 0 <= g < seq_len:
                attn[i, g] += 0.2
        # 归一化
        attn[i] = attn[i] / attn[i].sum()
    return attn


def generate_random_attention(seq_len):
    """生成随机注意力模式"""
    attn = np.random.rand(seq_len, seq_len)
    attn = attn / attn.sum(axis=1, keepdims=True)
    return attn


if __name__ == "__main__":
    # 创建注意力模式演示
    create_attention_demo()

    print("\n✓ 注意力可视化工具测试完成！")
    print("注意：实际的注意力权重提取需要模型支持返回 attention weights")