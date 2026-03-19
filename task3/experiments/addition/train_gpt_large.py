"""
实验2：大容量模型实验 - GPT

使用更大的模型配置：d_model=512, 更多层数，更长训练时间。
================================================================================
"""

import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.addition.train_gpt import *

def main():
    """主函数 - 使用大容量模型"""
    # 解析参数
    args = parse_args()

    # 设置大模型配置
    args.d_model = 512
    args.num_layers = 12
    args.dim_feedforward = 2048
    args.epochs = 60
    args.lr = 0.0001

    # 设置随机种子
    set_seed(args.seed)

    # 修改输出目录
    args.checkpoint_dir = 'outputs/experimental_results/exp2_capacity/models'
    args.log_dir = 'outputs/experimental_results/exp2_capacity/logs'

    # 打印配置
    print(f"\n实验2：大容量模型训练 - GPT")
    print("=" * 60)
    print(f"数据集类型: {args.dataset_type}")
    print(f"设备: {args.device}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print(f"模型配置:")
    print(f"  - d_model: {args.d_model}")
    print(f"  - num_layers: {args.num_layers}")
    print(f"  - dim_feedforward: {args.dim_feedforward}")
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

    # 额外保存大模型实验的结果
    exp_results = {
        'experiment': 'large_model',
        'model_type': 'gpt',
        'dataset_type': args.dataset_type,
        'results': results,
        'model_config': {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
        },
        'training_config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
        }
    }

    import json
    exp_results_path = os.path.join(args.log_dir, "gpt_large_results.json")
    with open(exp_results_path, 'w', encoding='utf-8') as f:
        json.dump(exp_results, f, indent=2, ensure_ascii=False)

    print(f"\n实验2完成！大容量 GPT 模型结果已保存")
    print("=" * 60)


if __name__ == "__main__":
    main()
