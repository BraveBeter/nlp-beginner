"""
批量训练脚本 - 运行所有实验组合
"""

import subprocess
import os

# 定义实验配置
model_types = ['softmax', 'perceptron']
data_files = ['bow_data.pkl', 'bigram_data.pkl', 'trigram_data.pkl']
learning_rates = [0.001, 0.01, 0.1]

# 训练参数
epochs = 50
batch_size = 32

print("=" * 70)
print("开始批量训练所有模型组合")
print("=" * 70)
print(f"模型类型: {model_types}")
print(f"数据文件: {data_files}")
print(f"学习率: {learning_rates}")
print(f"总训练数: {len(model_types) * len(data_files) * len(learning_rates)}")
print("=" * 70)

total = len(model_types) * len(data_files) * len(learning_rates)
current = 0

for model_type in model_types:
    for data_file in data_files:
        for lr in learning_rates:
            current += 1
            print(f"\n[{current}/{total}] 训练: {model_type} + {data_file} + lr={lr}")
            print("-" * 70)

            # 构建命令
            cmd = [
                'uv', 'run', 'train.py',
                '--model', model_type,
                '--data', data_file,
                '--lr', str(lr),
                '--epochs', str(epochs),
                '--batch_size', str(batch_size)
            ]

            # 运行训练
            try:
                result = subprocess.run(cmd, check=True, capture_output=False, text=True)
                print(f"✓ 完成: {model_type} + {data_file} + lr={lr}")
            except subprocess.CalledProcessError as e:
                print(f"✗ 失败: {model_type} + {data_file} + lr={lr}")
                print(f"错误: {e}")

print("\n" + "=" * 70)
print("所有训练完成!")
print("=" * 70)