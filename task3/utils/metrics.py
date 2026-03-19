"""
评估指标工具

实现常用的评估指标：
1. 准确率（Accuracy）：用于分类任务
2. 困惑度（Perplexity）：用于语言模型
"""

import torch
import torch.nn.functional as F
from typing import Optional


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_idx: Optional[int] = None
) -> float:
    """
    计算准确率

    Args:
        predictions: 预测结果 [batch_size, seq_len] 或 [batch_size]
        targets: 目标结果 [batch_size, seq_len] 或 [batch_size]
        ignore_idx: 忽略的索引（如 PAD token）

    Returns:
        准确率（0-1 之间的浮点数）

    示例:
        predictions = [1, 2, 3, 4, 5]
        targets = [1, 2, 3, 0, 5]  # 0 是 PAD token
        ignore_idx = 0

        准确率 = 4/4 = 1.0（忽略 PAD token）
    """
    # 确保形状一致
    assert predictions.shape == targets.shape, "预测和目标的形状不一致"

    # 展平张量
    predictions_flat = predictions.view(-1)
    targets_flat = targets.view(-1)

    # 如果需要忽略某些索引
    if ignore_idx is not None:
        mask = (targets_flat != ignore_idx)
        predictions_flat = predictions_flat[mask]
        targets_flat = targets_flat[mask]

    # 计算准确率
    correct = (predictions_flat == targets_flat).sum().item()
    total = targets_flat.size(0)

    accuracy = correct / total if total > 0 else 0.0

    return accuracy


def calculate_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_idx: Optional[int] = None
) -> float:
    """
    计算困惑度（Perplexity）

    困惑度是评估语言模型性能的重要指标。

    公式:
        PPL = exp(1/N * sum(-log(p(x_i))))

    其中:
        - N: token 的数量（不包括忽略的 token）
        - p(x_i): 第 i 个 token 的预测概率
        - log: 自然对数

    直观理解:
        - PPL = 1: 模型完全确定下一个 token
        - PPL = 10: 模型在 10 个等可能的候选中犹豫
        - PPL 越低，模型越好

    Args:
        logits: 模型输出 [batch_size, seq_len, vocab_size]
        targets: 目标索引 [batch_size, seq_len]
        ignore_idx: 忽略的索引（如 PAD token）

    Returns:
        困惑度（浮点数，>= 1）

    示例:
        logits: [batch_size=2, seq_len=3, vocab_size=1000]
        targets: [batch_size=2, seq_len=3]

        PPL = exp(cross_entropy_loss)
    """
    # 获取维度信息
    batch_size, seq_len, vocab_size = logits.shape

    # 展平
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # 计算交叉熵损失
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_idx if ignore_idx is not None else -100,
        reduction='mean'
    )

    # 困惑度 = exp(loss)
    perplexity = torch.exp(loss).item()

    return perplexity


def calculate_sequence_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_idx: Optional[int] = None
) -> float:
    """
    计算序列准确率（整个序列完全正确才算正确）

    用于加法任务等需要整个序列正确的场景。

    Args:
        predictions: 预测结果 [batch_size, seq_len]
        targets: 目标结果 [batch_size, seq_len]
        ignore_idx: 忽略的索引（如 PAD token）

    Returns:
        序列准确率（0-1 之间的浮点数）
    """
    batch_size = predictions.size(0)
    correct_sequences = 0

    for i in range(batch_size):
        pred = predictions[i]
        tgt = targets[i]

        # 如果需要忽略某些索引
        if ignore_idx is not None:
            mask = (tgt != ignore_idx)
            pred = pred[mask]
            tgt = tgt[mask]

        # 检查序列是否完全匹配
        if torch.equal(pred, tgt):
            correct_sequences += 1

    accuracy = correct_sequences / batch_size

    return accuracy


def print_statistics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    prefix: str = "",
    ignore_idx: Optional[int] = None
):
    """
    打印详细的统计信息

    Args:
        logits: 模型输出 [batch_size, seq_len, vocab_size]
        targets: 目标索引 [batch_size, seq_len]
        prefix: 打印前缀
        ignore_idx: 忽略的索引
    """
    # 获取预测
    predictions = torch.argmax(logits, dim=-1)

    # 计算各种指标
    token_acc = calculate_accuracy(predictions, targets, ignore_idx)
    seq_acc = calculate_sequence_accuracy(predictions, targets, ignore_idx)
    ppl = calculate_perplexity(logits, targets, ignore_idx)

    # 打印结果
    print(f"{prefix}Token 准确率: {token_acc:.4f}")
    print(f"{prefix}序列准确率: {seq_acc:.4f}")
    print(f"{prefix}困惑度: {ppl:.4f}")


# 测试
if __name__ == '__main__':
    print("测试评估指标工具")

    # 测试准确率
    print("\n" + "="*60)
    print("测试准确率计算")
    print("="*60)

    predictions = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 0, 4, 5],  # 第3个token错误
        [1, 2, 3, 4, 5],
    ])

    targets = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],  # 第3个token应该是3
        [1, 2, 3, 0, 0],  # 最后两个是PAD
    ])

    print(f"\n预测:")
    print(predictions)
    print(f"\n目标:")
    print(targets)

    # Token级别准确率（不忽略PAD）
    acc1 = calculate_accuracy(predictions, targets, ignore_idx=None)
    print(f"\nToken准确率（不忽略PAD）: {acc1:.4f}")

    # Token级别准确率（忽略PAD）
    acc2 = calculate_accuracy(predictions, targets, ignore_idx=0)
    print(f"Token准确率（忽略PAD）: {acc2:.4f}")

    # 序列准确率（忽略PAD）
    seq_acc = calculate_sequence_accuracy(predictions, targets, ignore_idx=0)
    print(f"序列准确率（忽略PAD）: {seq_acc:.4f}")

    # 测试困惑度
    print("\n" + "="*60)
    print("测试困惑度计算")
    print("="*60)

    batch_size = 2
    seq_len = 3
    vocab_size = 1000

    # 模拟模型输出
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # 模拟目标
    targets = torch.tensor([
        [10, 20, 30],
        [40, 50, 0],  # 最后一个是PAD
    ])

    # 计算困惑度
    ppl1 = calculate_perplexity(logits, targets, ignore_idx=None)
    ppl2 = calculate_perplexity(logits, targets, ignore_idx=0)

    print(f"\n困惑度（不忽略PAD）: {ppl1:.4f}")
    print(f"困惑度（忽略PAD）: {ppl2:.4f}")

    # 打印详细统计信息
    print("\n" + "="*60)
    print("测试详细统计信息")
    print("="*60)

    print_statistics(logits, targets, prefix="  ", ignore_idx=0)
