"""
GPT (Decoder-only) 训练脚本 - 加法任务

训练 Decoder-only GPT 架构来执行加法运算。
================================================================================
使用方法:
    python train_gpt.py --dataset_type iid --epochs 50 --batch_size 32

或者使用默认配置:
    python train_gpt.py
================================================================================
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Dict, Any
import json

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import GPT
from data.addition import AdditionTokenizer, AdditionDataGenerator, AdditionDataset, collate_fn_decoder_only, load_dataset_from_file
from utils import Trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练 GPT 模型')

    # 数据参数
    parser.add_argument('--dataset_type', type=str, default='iid',
                        choices=['iid', 'ood_length', 'ood_magnitude'],
                        help='数据集类型')
    parser.add_argument('--train_digits', type=int, nargs=2, default=[1, 3],
                        help='训练数据的位数范围 (最小, 最大)')
    parser.add_argument('--data_dir', type=str, default='outputs/data',
                        help='数据目录')
    parser.add_argument('--use_cached', action='store_true',
                        help='使用缓存的数据集')

    # 模型参数
    parser.add_argument('--d_model', type=int, default=256,
                        help='模型的维度')
    parser.add_argument('--nhead', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Transformer 层数')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 比率')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停耐心值')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪最大范数')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载工作进程数')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/models',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='outputs/logs',
                        help='日志保存目录')

    args = parser.parse_args()
    return args


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def load_data(args) -> tuple:
    """
    加载数据集

    Returns:
        (train_loader, val_loader, test_loader, tokenizer)
    """
    print(f"\n加载 {args.dataset_type} 数据集...")
    print("=" * 60)

    # 创建分词器
    tokenizer = AdditionTokenizer()

    # 数据集路径
    dataset_name = f"addition_{args.dataset_type}"
    train_path = os.path.join(args.data_dir, dataset_name, "train.pt")
    val_path = os.path.join(args.data_dir, dataset_name, "val.pt")
    test_path = os.path.join(args.data_dir, dataset_name, "test.pt")

    # 检查是否使用缓存数据
    if args.use_cached and os.path.exists(train_path):
        print(f"✓ 使用缓存数据集: {dataset_name}")

        train_dataset = load_dataset_from_file(train_path, mode='decoder_only')
        val_dataset = load_dataset_from_file(val_path, mode='decoder_only')
        test_dataset = load_dataset_from_file(test_path, mode='decoder_only')

    else:
        print(f"✓ 生成新的数据集: {dataset_name}")

        # 生成数据集
        generator = AdditionDataGenerator(tokenizer)

        if args.dataset_type == 'iid':
            # 生成完整数据集然后划分
            full_data = generator.generate_dataset(
                samples_per_type=50,
                digit_combinations=[(d1, d2) for d1 in range(args.train_digits[0], args.train_digits[1]+1)
                                   for d2 in range(args.train_digits[0], args.train_digits[1]+1)]
            )
            train_data, val_data, test_data = generator.split_iid(full_data)

        elif args.dataset_type == 'ood_length':
            # 按长度划分 - 训练集使用小位数，测试集使用大位数
            full_data = generator.generate_dataset(
                samples_per_type=50,
                digit_combinations=[(d1, d2) for d1 in range(1, 4) for d2 in range(1, 4)]
            )
            train_combinations = [(1, 1), (1, 2), (2, 1), (2, 2)]
            test_combinations = [(3, 3), (3, 4), (4, 3), (4, 4)]
            train_data, val_data, test_data = generator.split_ood_by_length(
                full_data, train_combinations, test_combinations
            )

        elif args.dataset_type == 'ood_magnitude':
            # 按数值划分 - 生成完整数据集
            full_data = generator.generate_dataset(
                samples_per_type=50,
                digit_combinations=[(d1, d2) for d1 in range(args.train_digits[0], args.train_digits[1]+1)
                                   for d2 in range(args.train_digits[0], args.train_digits[1]+1)]
            )
            # 使用数值划分
            train_data, val_data, test_data = generator.split_ood_by_magnitude(full_data)

        # 创建数据集
        train_dataset = AdditionDataset(train_data, tokenizer, mode='decoder_only')
        val_dataset = AdditionDataset(val_data, tokenizer, mode='decoder_only')
        test_dataset = AdditionDataset(test_data, tokenizer, mode='decoder_only')

        # 保存数据集
        os.makedirs(os.path.join(args.data_dir, dataset_name), exist_ok=True)
        torch.save(train_dataset, train_path)
        torch.save(val_dataset, val_path)
        torch.save(test_dataset, test_path)
        print(f"✓ 数据集已保存至: {os.path.join(args.data_dir, dataset_name)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_decoder_only,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_decoder_only,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_decoder_only,
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"批次数: {len(train_loader)}")

    return train_loader, val_loader, test_loader, tokenizer


def create_model(args, vocab_size: int) -> GPT:
    """
    创建 GPT 模型

    Args:
        args: 命令行参数
        vocab_size: 词表大小

    Returns:
        GPT 模型
    """
    print(f"\n创建 GPT 模型...")
    print("=" * 60)

    model = GPT(
        vocab_size=vocab_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.dim_feedforward,
        dropout=args.dropout,
    )

    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型配置:")
    print(f"  - d_model: {args.d_model}")
    print(f"  - nhead: {args.nhead}")
    print(f"  - num_layers: {args.num_layers}")
    print(f"  - dim_feedforward: {args.dim_feedforward}")
    print(f"  - dropout: {args.dropout}")
    print(f"总参数量: {num_params:,}")
    print(f"可训练参数量: {num_trainable_params:,}")

    return model


def train_model(args, model, train_loader, val_loader, tokenizer):
    """
    训练模型

    Args:
        args: 命令行参数
        model: GPT 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        tokenizer: 分词器

    Returns:
        训练历史
    """
    print(f"\n开始训练...")
    print("=" * 60)

    # 设备
    device = torch.device(args.device)

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_type="gpt",
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        pad_idx=tokenizer.pad_idx,
        max_grad_norm=args.max_grad_norm,
    )

    # 训练
    history = trainer.train(
        num_epochs=args.epochs,
        patience=args.patience,
        save_best_only=True,
    )

    return history, trainer


def evaluate_model(args, model, test_loader, tokenizer):
    """
    在测试集上评估模型

    Args:
        args: 命令行参数
        model: 训练好的模型
        test_loader: 测试数据加载器
        tokenizer: 分词器

    Returns:
        评估结果字典
    """
    print(f"\n在测试集上评估...")
    print("=" * 60)

    from utils import Evaluator

    device = torch.device(args.device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    evaluator = Evaluator(
        model=model,
        device=device,
        model_type="gpt",
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        pad_idx=tokenizer.pad_idx,
    )

    results = evaluator.evaluate(test_loader, criterion=criterion)

    print(f"\n测试集结果:")
    print(f"  - Loss: {results['test_loss']:.4f}")
    print(f"  - Token Accuracy: {results['token_accuracy']:.4f}")
    print(f"  - Sequence Accuracy: {results['sequence_accuracy']:.4f}")

    # 生成一些样本
    print(f"\n生成样本示例:")
    print("-" * 60)
    samples = evaluator.generate_samples(test_loader, num_samples=5)

    for i, (input_seq, target_seq, pred_seq) in enumerate(samples, 1):
        # 解码
        input_text = tokenizer.decode(input_seq, skip_special_tokens=False)
        target_text = tokenizer.decode(target_seq, skip_special_tokens=False)
        pred_text = tokenizer.decode(pred_seq, skip_special_tokens=False)

        print(f"\n样本 {i}:")
        print(f"  输入: {input_text}")
        print(f"  目标: {target_text}")
        print(f"  预测: {pred_text}")
        print(f"  正确: {target_text == pred_text}")

    return results


def save_config(args, results):
    """
    保存配置和结果

    Args:
        args: 命令行参数
        results: 评估结果
    """
    config = {
        'model_type': 'gpt',
        'dataset_type': args.dataset_type,
        'model_config': {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
        },
        'training_config': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'max_grad_norm': args.max_grad_norm,
        },
        'results': {
            'test_loss': results['test_loss'],
            'token_accuracy': results['token_accuracy'],
            'sequence_accuracy': results['sequence_accuracy'],
        },
    }

    config_path = os.path.join(args.log_dir, f"gpt_{args.dataset_type}_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 配置和结果已保存至: {config_path}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 打印配置
    print(f"\n训练配置")
    print("=" * 60)
    print(f"数据集类型: {args.dataset_type}")
    print(f"设备: {args.device}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print("=" * 60)

    # 加载数据
    train_loader, val_loader, test_loader, tokenizer = load_data(args)

    # 创建模型
    model = create_model(args, tokenizer.vocab_size)

    # 训练模型
    history, trainer = train_model(args, model, train_loader, val_loader, tokenizer)

    # 加载最佳模型
    best_model_path = os.path.join(args.checkpoint_dir, "gpt_best.pt")
    if os.path.exists(best_model_path):
        print(f"\n加载最佳模型: {best_model_path}")
        trainer.load_checkpoint(best_model_path)

    # 评估模型
    results = evaluate_model(args, model, test_loader, tokenizer)

    # 保存配置和结果
    save_config(args, results)

    print(f"\n训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()