"""
数据处理流程完整测试脚本

测试从数据生成到训练准备的全流程。
"""

import torch
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data.addition import (
    AdditionTokenizer,
    AdditionDataGenerator,
    AdditionDataset,
    collate_fn_encoder_decoder,
    collate_fn_decoder_only,
    load_dataset_from_file
)
from utils import (
    generate_padding_mask,
    generate_square_subsequent_mask,
    calculate_accuracy,
    calculate_perplexity
)


def test_data_pipeline():
    """测试完整的数据处理流程"""
    print("\n" + "="*80)
    print("测试完整的数据处理流程")
    print("="*80)

    # 1. 创建分词器
    print("\n[1/6] 创建分词器...")
    tokenizer = AdditionTokenizer()
    print(f"✓ 词表大小: {tokenizer.vocab_size}")

    # 2. 生成数据集
    print("\n[2/6] 生成数据集...")
    generator = AdditionDataGenerator(
        min_digits=1,
        max_digits=3,
        train_split=0.8,
        seed=42
    )

    dataset = generator.generate_dataset(
        samples_per_type=50,
        digit_combinations=[(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
    )

    # IID 划分
    train_data, val_data, test_data = generator.split_iid(dataset, val_ratio=0.1)
    print(f"✓ 训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

    # 3. 创建 Dataset
    print("\n[3/6] 创建 Dataset...")

    # Encoder-Decoder 模式
    dataset_enc_dec = AdditionDataset(train_data, tokenizer, mode='encoder_decoder')
    print(f"✓ Encoder-Decoder 模式: {len(dataset_enc_dec)} 样本")

    # Decoder-only 模式
    dataset_dec_only = AdditionDataset(train_data, tokenizer, mode='decoder_only')
    print(f"✓ Decoder-only 模式: {len(dataset_dec_only)} 样本")

    # 4. 创建 DataLoader
    print("\n[4/6] 创建 DataLoader...")

    # Encoder-Decoder
    batch_enc_dec = [dataset_enc_dec[i] for i in range(4)]
    batch_enc_dec_collated = collate_fn_encoder_decoder(batch_enc_dec)
    print(f"✓ Encoder-Decoder batch shape:")
    print(f"  源序列: {batch_enc_dec_collated['src'].shape}")
    print(f"  目标序列: {batch_enc_dec_collated['tgt'].shape}")

    # Decoder-only
    batch_dec_only = [dataset_dec_only[i] for i in range(4)]
    batch_dec_only_collated = collate_fn_decoder_only(batch_dec_only)
    print(f"✓ Decoder-only batch shape:")
    print(f"  输入: {batch_dec_only_collated['input'].shape}")

    # 5. 生成掩码
    print("\n[5/6] 生成掩码...")

    # Padding mask
    src = batch_enc_dec_collated['src']
    tgt = batch_enc_dec_collated['tgt']

    src_padding_mask = generate_padding_mask(src, pad_idx=tokenizer.pad_idx)
    tgt_padding_mask = generate_padding_mask(tgt, pad_idx=tokenizer.pad_idx)

    print(f"✓ Padding mask:")
    print(f"  源序列: {src_padding_mask.shape}")
    print(f"  目标序列: {tgt_padding_mask.shape}")

    # Subsequent mask (因果掩码)
    tgt_len = tgt.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_len)
    print(f"✓ Subsequent mask: {tgt_mask.shape}")

    # 6. 测试评估指标
    print("\n[6/6] 测试评估指标...")

    # 模拟预测
    predictions = torch.randint(0, tokenizer.vocab_size, tgt.shape)
    targets = tgt

    # Token 准确率
    token_acc = calculate_accuracy(predictions, targets, ignore_idx=tokenizer.pad_idx)
    print(f"✓ Token 准确率: {token_acc:.4f}")

    # 序列准确率
    seq_acc = calculate_sequence_accuracy(predictions, targets, ignore_idx=tokenizer.pad_idx)
    print(f"✓ 序列准确率: {seq_acc:.4f}")

    # 模拟 logits 用于计算困惑度
    batch_size, seq_len = tgt.shape
    logits = torch.randn(batch_size, seq_len, tokenizer.vocab_size)
    ppl = calculate_perplexity(logits, targets, ignore_idx=tokenizer.pad_idx)
    print(f"✓ 困惑度: {ppl:.4f}")

    print("\n" + "="*80)
    print("✓ 数据处理流程测试完成！")
    print("="*80)

    return True


def test_data_loading():
    """测试从文件加载数据"""
    print("\n" + "="*80)
    print("测试从文件加载数据")
    print("="*80)

    # 检查文件是否存在
    data_dir = Path('outputs/data/addition_iid')
    if not data_dir.exists():
        print("✗ 数据集不存在，请先运行数据生成器")
        return False

    # 创建分词器
    tokenizer = AdditionTokenizer()

    # 加载训练集
    print("\n[1/3] 加载训练集...")
    train_dataset = load_dataset_from_file(
        data_dir / 'train.txt',
        tokenizer,
        mode='encoder_decoder'
    )
    print(f"✓ 训练集: {len(train_dataset)} 样本")

    # 加载验证集
    print("\n[2/3] 加载验证集...")
    val_dataset = load_dataset_from_file(
        data_dir / 'val.txt',
        tokenizer,
        mode='encoder_decoder'
    )
    print(f"✓ 验证集: {len(val_dataset)} 样本")

    # 加载测试集
    print("\n[3/3] 加载测试集...")
    test_dataset = load_dataset_from_file(
        data_dir / 'test.txt',
        tokenizer,
        mode='encoder_decoder'
    )
    print(f"✓ 测试集: {len(test_dataset)} 样本")

    # 获取一个样本
    sample = train_dataset[0]
    print(f"\n样本示例:")
    print(f"  源序列: {sample['src_text']}")
    print(f"  目标序列: {sample['tgt_text']}")

    print("\n" + "="*80)
    print("✓ 数据加载测试完成！")
    print("="*80)

    return True


def calculate_sequence_accuracy(predictions, targets, ignore_idx=None):
    """计算序列准确率"""
    batch_size = predictions.size(0)
    correct_sequences = 0

    for i in range(batch_size):
        pred = predictions[i]
        tgt = targets[i]

        if ignore_idx is not None:
            mask = (tgt != ignore_idx)
            pred = pred[mask]
            tgt = tgt[mask]

        if torch.equal(pred, tgt):
            correct_sequences += 1

    return correct_sequences / batch_size


if __name__ == '__main__':
    # 测试数据处理流程
    success1 = test_data_pipeline()

    # 测试数据加载
    success2 = test_data_loading()

    if success1 and success2:
        print("\n" + "="*80)
        print("✓ 所有测试通过！数据处理模块已准备好用于训练。")
        print("="*80)
    else:
        print("\n✗ 部分测试失败")
