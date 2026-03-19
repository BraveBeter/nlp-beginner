"""
训练基础设施测试脚本

测试 Phase 3 的训练、评估和对比功能。
================================================================================
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Transformer, GPT
from data.addition import AdditionTokenizer, AdditionDataset, collate_fn_encoder_decoder, collate_fn_decoder_only
from utils import Trainer, Evaluator


def create_dummy_data(num_samples=100):
    """创建虚拟数据用于测试"""
    tokenizer = AdditionTokenizer()

    # 创建简单的加法数据
    data = []
    for i in range(num_samples):
        a = i % 10
        b = (i // 10) % 10
        result = a + b
        formula = f"{a}+{b}={result}"
        data.append(formula)

    return data, tokenizer


def test_transformer_training():
    """测试 Transformer 训练"""
    print("\n" + "=" * 60)
    print("测试 Transformer 训练")
    print("=" * 60)

    # 创建数据
    data, tokenizer = create_dummy_data(num_samples=100)

    # 划分数据集
    train_data = data[:80]
    val_data = data[80:90]
    test_data = data[90:]

    # 创建数据集
    train_dataset = AdditionDataset(train_data, tokenizer, mode='encoder_decoder')
    val_dataset = AdditionDataset(val_data, tokenizer, mode='encoder_decoder')
    test_dataset = AdditionDataset(test_data, tokenizer, mode='encoder_decoder')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn_encoder_decoder)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn_encoder_decoder)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn_encoder_decoder)

    print(f"✓ 数据加载完成")
    print(f"  - 训练集: {len(train_dataset)} 样本")
    print(f"  - 验证集: {len(val_dataset)} 样本")
    print(f"  - 测试集: {len(test_dataset)} 样本")

    # 创建模型
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=0.1,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型创建完成 (参数量: {num_params:,})")

    # 训练设置
    device = torch.device('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_type="transformer",
        checkpoint_dir="outputs/test_models",
        log_dir="outputs/test_logs",
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        pad_idx=tokenizer.pad_idx,
        max_grad_norm=1.0,
    )

    print(f"✓ 训练器创建完成")

    # 训练几个 epoch
    print(f"\n开始训练...")
    history = trainer.train(num_epochs=3, patience=5, save_best_only=True)

    print(f"✓ 训练完成")
    print(f"  - 最终训练损失: {history['train_losses'][-1]:.4f}")
    print(f"  - 最终验证损失: {history['val_losses'][-1]:.4f}")

    # 评估
    print(f"\n开始评估...")
    evaluator = Evaluator(
        model=model,
        device=device,
        model_type="transformer",
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        pad_idx=tokenizer.pad_idx,
    )

    results = evaluator.evaluate(test_loader, criterion=criterion)
    print(f"✓ 评估完成")
    print(f"  - 测试损失: {results['test_loss']:.4f}")
    print(f"  - Token 准确率: {results['token_accuracy']:.4f}")
    print(f"  - 序列准确率: {results['sequence_accuracy']:.4f}")

    return True


def test_gpt_training():
    """测试 GPT 训练"""
    print("\n" + "=" * 60)
    print("测试 GPT 训练")
    print("=" * 60)

    # 创建数据
    data, tokenizer = create_dummy_data(num_samples=100)

    # 划分数据集
    train_data = data[:80]
    val_data = data[80:90]
    test_data = data[90:]

    # 创建数据集
    train_dataset = AdditionDataset(train_data, tokenizer, mode='decoder_only')
    val_dataset = AdditionDataset(val_data, tokenizer, mode='decoder_only')
    test_dataset = AdditionDataset(test_data, tokenizer, mode='decoder_only')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn_decoder_only)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn_decoder_only)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn_decoder_only)

    print(f"✓ 数据加载完成")
    print(f"  - 训练集: {len(train_dataset)} 样本")
    print(f"  - 验证集: {len(val_dataset)} 样本")
    print(f"  - 测试集: {len(test_dataset)} 样本")

    # 创建模型
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        num_layers=4,
        d_model=128,
        nhead=4,
        d_ff=256,
        dropout=0.1,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型创建完成 (参数量: {num_params:,})")

    # 训练设置
    device = torch.device('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx, reduction='none')

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_type="gpt",
        checkpoint_dir="outputs/test_models",
        log_dir="outputs/test_logs",
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        pad_idx=tokenizer.pad_idx,
        max_grad_norm=1.0,
    )

    print(f"✓ 训练器创建完成")

    # 训练几个 epoch
    print(f"\n开始训练...")
    history = trainer.train(num_epochs=3, patience=5, save_best_only=True)

    print(f"✓ 训练完成")
    print(f"  - 最终训练损失: {history['train_losses'][-1]:.4f}")
    print(f"  - 最终验证损失: {history['val_losses'][-1]:.4f}")

    # 评估
    print(f"\n开始评估...")
    evaluator = Evaluator(
        model=model,
        device=device,
        model_type="gpt",
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        pad_idx=tokenizer.pad_idx,
    )

    results = evaluator.evaluate(test_loader, criterion=criterion)
    print(f"✓ 评估完成")
    print(f"  - 测试损失: {results['test_loss']:.4f}")
    print(f"  - Token 准确率: {results['token_accuracy']:.4f}")
    print(f"  - 序列准确率: {results['sequence_accuracy']:.4f}")

    return True


def test_model_comparison():
    """测试模型对比"""
    print("\n" + "=" * 60)
    print("测试模型对比")
    print("=" * 60)

    # 创建数据
    data, tokenizer = create_dummy_data(num_samples=50)

    # 创建测试数据集
    test_dataset = AdditionDataset(data, tokenizer, mode='encoder_decoder')
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn_encoder_decoder)

    # 创建两个模型
    model1 = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=0.1,
    )

    model2 = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,  # 更深的网络
        num_decoder_layers=3,
        d_ff=256,
        dropout=0.1,
    )

    print(f"✓ 模型创建完成")

    # 设备
    device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    # 对比
    from utils import compare_models

    results = compare_models(
        model1, model2, test_loader, device,
        model1_name="Transformer (2 layers)",
        model2_name="Transformer (3 layers)",
        model_type="transformer",
    )

    print(f"✓ 对比完成")

    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Phase 3 训练基础设施测试")
    print("=" * 60)

    try:
        # 测试 Transformer 训练
        test_transformer_training()

        # 测试 GPT 训练
        test_gpt_training()

        # 测试模型对比
        test_model_comparison()

        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)