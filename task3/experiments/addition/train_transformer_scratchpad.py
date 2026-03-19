"""
实验3：Scratchpad（草稿纸）技术实验 - Transformer

训练模型学习逐步推理，使用包含中间步骤的数据格式。
例如: "123+456= 9,7,5→579"
================================================================================
"""

import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.addition.train_transformer import *

def main():
    """主函数 - 使用 scratchpad 格式"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 修改输出目录
    args.checkpoint_dir = 'outputs/experimental_results/exp3_scratchpad/models'
    args.log_dir = 'outputs/experimental_results/exp3_scratchpad/logs'

    # 打印配置
    print(f"\n实验3：Scratchpad 技术训练 - Transformer")
    print("=" * 60)
    print(f"数据集类型: {args.dataset_type}")
    print(f"设备: {args.device}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print(f"Scratchpad 格式: 启用（包含中间推理步骤）")
    print("=" * 60)

    # 加载数据 - 使用 scratchpad 格式
    print(f"\n加载 {args.dataset_type} 数据集（Scratchpad 格式）...")
    print("=" * 60)

    # 创建分词器
    tokenizer = AdditionTokenizer()

    # 生成 scratchpad 格式的数据集
    generator = AdditionDataGenerator()

    if args.dataset_type == 'iid':
        # 生成完整数据集然后划分，使用 scratchpad
        full_data = generator.generate_dataset(
            samples_per_type=50,
            digit_combinations=[(d1, d2) for d1 in range(args.train_digits[0], args.train_digits[1]+1)
                               for d2 in range(args.train_digits[0], args.train_digits[1]+1)],
            scratchpad=True  # 启用 scratchpad
        )
        train_data, val_data, test_data = generator.split_iid(full_data)

    # 创建数据集
    train_dataset = AdditionDataset(train_data, tokenizer, mode='encoder_decoder')
    val_dataset = AdditionDataset(val_data, tokenizer, mode='encoder_decoder')
    test_dataset = AdditionDataset(test_data, tokenizer, mode='encoder_decoder')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_encoder_decoder,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_encoder_decoder,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_encoder_decoder,
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 显示一些样本
    print(f"\nScratchpad 格式样本示例:")
    for i in range(min(3, len(train_data))):
        print(f"  {i+1}. {train_data[i]}")

    # 创建模型
    model = create_model(args, tokenizer.vocab_size)

    # 训练模型
    history, trainer = train_model(args, model, train_loader, val_loader, tokenizer)

    # 加载最佳模型
    best_model_path = os.path.join(args.checkpoint_dir, "transformer_best.pt")
    if os.path.exists(best_model_path):
        print(f"\n加载最佳模型: {best_model_path}")
        trainer.load_checkpoint(best_model_path)

    # 评估模型
    results = evaluate_model(args, model, test_loader, tokenizer)

    # 保存配置和结果
    save_config(args, results)

    # 额外保存 scratchpad 实验的结果
    exp_results = {
        'experiment': 'scratchpad',
        'model_type': 'transformer',
        'dataset_type': args.dataset_type,
        'scratchpad': True,
        'results': results,
        'model_config': {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_encoder_layers': args.num_encoder_layers,
            'num_decoder_layers': args.num_decoder_layers,
        },
        'training_config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
        }
    }

    import json
    exp_results_path = os.path.join(args.log_dir, "transformer_scratchpad_results.json")
    with open(exp_results_path, 'w', encoding='utf-8') as f:
        json.dump(exp_results, f, indent=2, ensure_ascii=False)

    print(f"\n实验3完成！Scratchpad 训练的 Transformer 模型结果已保存")
    print("=" * 60)


if __name__ == "__main__":
    main()
