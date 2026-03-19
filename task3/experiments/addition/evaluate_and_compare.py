"""
模型评估和对比脚本

用于评估和对比不同模型在加法任务上的性能。
================================================================================
功能:
1. 在测试集上评估单个模型
2. 对比不同架构 (Transformer vs GPT)
3. 对比 IID vs OOD 性能
4. 生成详细的评估报告和可视化
================================================================================
使用方法:
    # 评估单个模型
    python evaluate_and_compare.py --model_type transformer --dataset_type iid

    # 对比两个模型
    python evaluate_and_compare.py --mode compare --model1 transformer_iid --model2 gpt_iid

    # 评估所有模型
    python evaluate_and_compare.py --mode evaluate_all
================================================================================
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import Transformer, GPT
from data.addition import AdditionTokenizer, AdditionDataset, collate_fn_encoder_decoder, collate_fn_decoder_only, load_dataset_from_file
from utils import Evaluator, compare_models


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估和对比模型')

    parser.add_argument('--mode', type=str, default='evaluate',
                        choices=['evaluate', 'compare', 'evaluate_all'],
                        help='运行模式')

    # 评估模式参数
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'gpt'],
                        help='模型类型')
    parser.add_argument('--dataset_type', type=str, default='iid',
                        choices=['iid', 'ood_length', 'ood_magnitude'],
                        help='数据集类型')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='模型检查点路径')

    # 对比模式参数
    parser.add_argument('--model1', type=str, default=None,
                        help='第一个模型名称 (如: transformer_iid)')
    parser.add_argument('--model2', type=str, default=None,
                        help='第二个模型名称 (如: gpt_iid)')

    # 数据和模型参数
    parser.add_argument('--data_dir', type=str, default='outputs/data',
                        help='数据目录')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/models',
                        help='检查点目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='评估结果输出目录')

    args = parser.parse_args()
    return args


def load_model_and_tokenizer(checkpoint_path: str, model_type: str, device: torch.device) -> Tuple[Any, AdditionTokenizer]:
    """
    加载模型和分词器

    Args:
        checkpoint_path: 检查点文件路径
        model_type: 模型类型 ('transformer' or 'gpt')
        device: 计算设备

    Returns:
        (模型, 分词器)
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 创建分词器
    tokenizer = AdditionTokenizer()

    # 创建模型
    if model_type == 'transformer':
        model = Transformer(
            vocab_size=tokenizer.vocab_size,
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            d_ff=1024,
            dropout=0.1,
        )
    elif model_type == 'gpt':
        model = GPT(
            vocab_size=tokenizer.vocab_size,
            num_layers=6,
            d_model=256,
            nhead=8,
            d_ff=1024,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ 加载模型: {checkpoint_path}")
    print(f"  - 模型类型: {model_type}")
    print(f"  - 训练轮数: {checkpoint['epoch']}")
    print(f"  - 最佳验证损失: {checkpoint['best_val_loss']:.4f}")

    return model, tokenizer


def load_test_dataset(dataset_type: str, data_dir: str, tokenizer: AdditionTokenizer, model_type: str) -> DataLoader:
    """
    加载测试数据集

    Args:
        dataset_type: 数据集类型
        data_dir: 数据目录
        tokenizer: 分词器
        model_type: 模型类型

    Returns:
        测试数据加载器
    """
    dataset_name = f"addition_{dataset_type}"
    test_path = os.path.join(data_dir, dataset_name, "test.pt")

    # 确定模式
    mode = 'encoder_decoder' if model_type == 'transformer' else 'decoder_only'

    # 加载数据集
    test_dataset = load_dataset_from_file(test_path, tokenizer=tokenizer, mode=mode)

    # 选择 collate_fn
    collate_fn = collate_fn_encoder_decoder if model_type == 'transformer' else collate_fn_decoder_only

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"✓ 加载测试集: {dataset_name}")
    print(f"  - 样本数: {len(test_dataset)}")
    print(f"  - 模式: {mode}")

    return test_loader


def evaluate_single_model(args) -> Dict[str, float]:
    """
    评估单个模型

    Args:
        args: 命令行参数

    Returns:
        评估结果字典
    """
    print(f"\n评估模型: {args.model_type} on {args.dataset_type}")
    print("=" * 60)

    # 确定检查点路径
    if args.checkpoint_path is None:
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"{args.model_type}_best.pt"
        )
    else:
        checkpoint_path = args.checkpoint_path

    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    # 加载模型和分词器
    device = torch.device(args.device)
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, args.model_type, device)

    # 加载测试集
    test_loader = load_test_dataset(args.dataset_type, args.data_dir, tokenizer, args.model_type)

    # 评估
    evaluator = Evaluator(
        model=model,
        device=device,
        model_type=args.model_type,
        bos_idx=tokenizer.bos_idx,
        eos_idx=tokenizer.eos_idx,
        pad_idx=tokenizer.pad_idx,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx, reduction='mean')
    results = evaluator.evaluate(test_loader, criterion=criterion)

    # 打印结果
    print(f"\n评估结果:")
    print(f"  - Test Loss: {results['test_loss']:.4f}")
    print(f"  - Token Accuracy: {results['token_accuracy']:.4f}")
    print(f"  - Sequence Accuracy: {results['sequence_accuracy']:.4f}")

    # 生成样本
    print(f"\n生成样本示例:")
    print("-" * 60)
    samples = evaluator.generate_samples(test_loader, num_samples=10)

    correct = 0
    for i, (item1, item2, pred) in enumerate(samples, 1):
        if args.model_type == 'transformer':
            src_text = tokenizer.decode(item1, skip_special_tokens=False)
            tgt_text = tokenizer.decode(item2, skip_special_tokens=False)
            pred_text = tokenizer.decode(pred, skip_special_tokens=False)

            is_correct = (tgt_text == pred_text)
            if is_correct:
                correct += 1

            print(f"{i}. {src_text} → {pred_text} {'✓' if is_correct else '✗'}")
        else:
            input_text = tokenizer.decode(item1, skip_special_tokens=False)
            target_text = tokenizer.decode(item2, skip_special_tokens=False)
            pred_text = tokenizer.decode(pred, skip_special_tokens=False)

            is_correct = (target_text == pred_text)
            if is_correct:
                correct += 1

            print(f"{i}. {input_text} → {pred_text} {'✓' if is_correct else '✗'}")

    print(f"\n样本准确率: {correct}/{len(samples)} = {correct/len(samples):.4f}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(
        args.output_dir,
        f"{args.model_type}_{args.dataset_type}_results.json"
    )

    # 转换结果为可序列化格式
    results_serializable = {
        'model_type': args.model_type,
        'dataset_type': args.dataset_type,
        'checkpoint_path': checkpoint_path,
        'test_loss': float(results['test_loss']) if results['test_loss'] is not None else None,
        'token_accuracy': float(results['token_accuracy']),
        'sequence_accuracy': float(results['sequence_accuracy']),
        'sample_accuracy': float(correct / len(samples)),
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存至: {result_path}")

    return results_serializable


def compare_two_models(args) -> Dict[str, Dict[str, float]]:
    """
    对比两个模型

    Args:
        args: 命令行参数

    Returns:
        对比结果字典
    """
    print(f"\n对比模型: {args.model1} vs {args.model2}")
    print("=" * 60)

    # 解析模型名称
    model1_type, model1_dataset = args.model1.split('_')
    model2_type, model2_dataset = args.model2.split('_')

    # 加载模型1
    checkpoint1 = os.path.join(args.checkpoint_dir, f"{model1_type}_best.pt")
    device = torch.device(args.device)
    model1, tokenizer1 = load_model_and_tokenizer(checkpoint1, model1_type, device)
    test_loader1 = load_test_dataset(model1_dataset, args.data_dir, tokenizer1, model1_type)

    # 加载模型2
    checkpoint2 = os.path.join(args.checkpoint_dir, f"{model2_type}_best.pt")
    model2, tokenizer2 = load_model_and_tokenizer(checkpoint2, model2_type, device)
    test_loader2 = load_test_dataset(model2_dataset, args.data_dir, tokenizer2, model2_type)

    # 对比
    results = compare_models(
        model1, model2, test_loader1, device,
        model1_name=args.model1,
        model2_name=args.model2,
        model_type=model1_type,
    )

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, f"comparison_{args.model1}_vs_{args.model2}.json")

    # 转换结果为可序列化格式
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = {
            'test_loss': float(value['test_loss']) if value['test_loss'] is not None else None,
            'token_accuracy': float(value['token_accuracy']),
            'sequence_accuracy': float(value['sequence_accuracy']),
        }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 对比结果已保存至: {result_path}")

    return results_serializable


def evaluate_all_models(args) -> Dict[str, Dict[str, float]]:
    """
    评估所有模型组合

    Args:
        args: 命令行参数

    Returns:
        所有评估结果字典
    """
    print(f"\n评估所有模型组合")
    print("=" * 60)

    model_types = ['transformer', 'gpt']
    dataset_types = ['iid', 'ood_length', 'ood_magnitude']

    all_results = {}

    for model_type in model_types:
        all_results[model_type] = {}

        for dataset_type in dataset_types:
            print(f"\n评估: {model_type} on {dataset_type}")
            print("-" * 60)

            try:
                # 检查检查点是否存在
                checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_type}_best.pt")
                if not os.path.exists(checkpoint_path):
                    print(f"✗ 检查点不存在: {checkpoint_path}")
                    all_results[model_type][dataset_type] = None
                    continue

                # 更新 args
                args.model_type = model_type
                args.dataset_type = dataset_type
                args.checkpoint_path = checkpoint_path

                # 评估
                results = evaluate_single_model(args)
                all_results[model_type][dataset_type] = results

            except Exception as e:
                print(f"✗ 评估失败: {e}")
                all_results[model_type][dataset_type] = None

    # 生成汇总报告
    print(f"\n汇总报告")
    print("=" * 60)

    # 创建 DataFrame
    data = []
    for model_type in model_types:
        for dataset_type in dataset_types:
            if all_results[model_type][dataset_type] is not None:
                result = all_results[model_type][dataset_type]
                data.append({
                    'Model': model_type,
                    'Dataset': dataset_type,
                    'Token Acc': result['token_accuracy'],
                    'Sequence Acc': result['sequence_accuracy'],
                })

    df = pd.DataFrame(data)

    # 打印表格
    print("\nToken Accuracy:")
    print(df.pivot(index='Model', columns='Dataset', values='Token Acc'))

    print("\nSequence Accuracy:")
    print(df.pivot(index='Model', columns='Dataset', values='Sequence Acc'))

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "all_models_evaluation.json")

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 所有结果已保存至: {result_path}")

    # 生成可视化
    generate_comparison_plots(all_results, args.output_dir)

    return all_results


def generate_comparison_plots(results: Dict, output_dir: str):
    """
    生成对比图表

    Args:
        results: 所有评估结果
        output_dir: 输出目录
    """
    print(f"\n生成对比图表...")

    # 准备数据
    data = []
    for model_type in results.keys():
        for dataset_type in results[model_type].keys():
            if results[model_type][dataset_type] is not None:
                result = results[model_type][dataset_type]
                data.append({
                    'Model': model_type.upper(),
                    'Dataset': dataset_type.replace('_', ' ').title(),
                    'Token Accuracy': result['token_accuracy'],
                    'Sequence Accuracy': result['sequence_accuracy'],
                })

    df = pd.DataFrame(data)

    # 设置图表风格
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Token Accuracy 对比
    ax1 = axes[0]
    sns.barplot(data=df, x='Dataset', y='Token Accuracy', hue='Model', ax=ax1)
    ax1.set_title('Token Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('Token Accuracy', fontsize=12)
    ax1.legend(title='Model', fontsize=10)
    ax1.set_ylim(0, 1.0)

    # Sequence Accuracy 对比
    ax2 = axes[1]
    sns.barplot(data=df, x='Dataset', y='Sequence Accuracy', hue='Model', ax=ax2)
    ax2.set_title('Sequence Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.set_ylabel('Sequence Accuracy', fontsize=12)
    ax2.legend(title='Model', fontsize=10)
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()

    # 保存图表
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图表已保存至: {plot_path}")

    plt.close()


def main():
    """主函数"""
    args = parse_args()

    print(f"\n模型评估和对比工具")
    print("=" * 60)
    print(f"运行模式: {args.mode}")
    print(f"计算设备: {args.device}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    if args.mode == 'evaluate':
        # 评估单个模型
        evaluate_single_model(args)

    elif args.mode == 'compare':
        # 对比两个模型
        if args.model1 is None or args.model2 is None:
            raise ValueError("--model1 和 --model2 是对比模式必需的参数")
        compare_two_models(args)

    elif args.mode == 'evaluate_all':
        # 评估所有模型
        evaluate_all_models(args)

    print(f"\n评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()