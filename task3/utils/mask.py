"""
掩码生成工具

实现各种掩码的生成，用于 Transformer 的注意力机制。

支持的掩码类型：
1. Padding mask：屏蔽填充 token
2. Subsequent mask (Causal mask)：因果掩码，用于 Decoder 的自注意力
"""

import torch
from typing import Optional


def generate_padding_mask(
    sequences: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    生成填充掩码

    标记出哪些位置是填充（PAD），需要被屏蔽。

    Args:
        sequences: 序列张量 [batch_size, seq_len]
        pad_idx: 填充 token 的索引

    Returns:
        掩码张量 [batch_size, seq_len]
        True 表示该位置是填充，需要屏蔽
        False 表示该位置是有效 token

    示例:
        sequences = [[1, 2, 3, 0, 0],
                    [4, 5, 0, 0, 0]]
        pad_idx = 0

        返回: [[False, False, False, True, True],
               [False, False, True, True, True]]
    """
    mask = (sequences == pad_idx)
    return mask


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    生成因果掩码（上三角矩阵）

    用于 Decoder 的自注意力，防止当前位置看到未来的信息。

    Args:
        sz: 序列长度

    Returns:
        掩码张量 [sz, sz]
        True 表示需要屏蔽的位置（上三角，不包括对角线）
        False 表示保留的位置（下三角，包括对角线）

    示例:
        sz = 4

        返回: [[False, True,  True,  True ],
               [False, False, True,  True ],
               [False, False, False, True ],
               [False, False, False, False]]

    直观理解:
        - 位置 0 只能看到自己
        - 位置 1 可以看到位置 0 和 1
        - 位置 2 可以看到位置 0, 1, 2
        - 位置 3 可以看到位置 0, 1, 2, 3
    """
    # 生成上三角矩阵（不包括对角线）
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
    return mask


def combine_masks(
    padding_mask: Optional[torch.Tensor] = None,
    subsequent_mask: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """
    组合填充掩码和因果掩码

    Args:
        padding_mask: 填充掩码 [batch_size, seq_len]
        subsequent_mask: 因果掩码 [seq_len, seq_len]

    Returns:
        组合后的掩码 [batch_size, seq_len, seq_len]
        或者 None（如果两个掩码都为 None）

    说明:
        如果只有填充掩码，扩展为 [batch_size, 1, seq_len]
        如果只有因果掩码，扩展为 [1, seq_len, seq_len]
        如果两个都有，组合为 [batch_size, seq_len, seq_len]
    """
    if padding_mask is None and subsequent_mask is None:
        return None

    batch_size = 1
    seq_len = 1

    # 确定维度
    if padding_mask is not None:
        batch_size = padding_mask.size(0)
        seq_len = padding_mask.size(1)
    elif subsequent_mask is not None:
        seq_len = subsequent_mask.size(0)

    # 初始化组合掩码
    combined_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)

    # 应用填充掩码
    if padding_mask is not None:
        # padding_mask: [batch_size, seq_len]
        # 扩展为 [batch_size, seq_len, seq_len]
        combined_mask = combined_mask | padding_mask.unsqueeze(1)

    # 应用因果掩码
    if subsequent_mask is not None:
        # subsequent_mask: [seq_len, seq_len]
        # 扩展为 [batch_size, seq_len, seq_len]
        combined_mask = combined_mask | subsequent_mask.unsqueeze(0)

    return combined_mask


# 测试
if __name__ == '__main__':
    print("测试掩码生成工具")

    # 测试填充掩码
    print("\n" + "="*60)
    print("测试填充掩码 (Padding Mask)")
    print("="*60)

    sequences = torch.tensor([
        [1, 2, 3, 0, 0],
        [4, 5, 0, 0, 0],
        [6, 7, 8, 9, 10],
    ])

    print(f"\n输入序列:")
    print(sequences)

    padding_mask = generate_padding_mask(sequences, pad_idx=0)
    print(f"\n填充掩码 (True=PAD):")
    print(padding_mask)

    # 测试因果掩码
    print("\n" + "="*60)
    print("测试因果掩码 (Subsequent Mask)")
    print("="*60)

    sz = 5
    subsequent_mask = generate_square_subsequent_mask(sz)

    print(f"\n因果掩码 (sz={sz}, True=屏蔽):")
    print(subsequent_mask)

    # 测试组合掩码
    print("\n" + "="*60)
    print("测试组合掩码")
    print("="*60)

    combined = combine_masks(padding_mask, subsequent_mask)
    print(f"\n组合掩码 shape: {combined.shape}")
    print(f"第一个样本的组合掩码:")
    print(combined[0])

    # 可视化掩码
    print("\n" + "="*60)
    print("可视化掩码")
    print("="*60)

    def visualize_mask(mask, title):
        print(f"\n{title}:")
        for row in mask:
            row_str = " ".join(["█" if v else " " for v in row])
            print(f"|{row_str}|")

    # 可视化因果掩码
    visualize_mask(subsequent_mask.numpy(), "因果掩码 (█=屏蔽)")

    # 可视化填充掩码
    print(f"\n填充掩码:")
    for i, row in enumerate(padding_mask.numpy()):
        row_str = " ".join(["PAD" if v else str(sequences[i, j].item()) for j, v in enumerate(row)])
        print(f"样本 {i}: {row_str}")
