"""
模型结构可视化脚本

展示 Transformer 和 GPT 模型的结构和参数统计。
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.transformer import Transformer
from models.gpt import GPT


def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def show_transformer_architecture():
    """展示 Transformer 模型结构"""
    print("\n" + "="*80)
    print("Transformer (Encoder-Decoder) 架构")
    print("="*80)

    model = Transformer(
        vocab_size=1000,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1
    )

    total_params, trainable_params = count_parameters(model)

    print("\n模型配置:")
    print(f"  词表大小: 1000")
    print(f"  隐藏维度: 512")
    print(f"  注意力头数: 8")
    print(f"  编码器层数: 6")
    print(f"  解码器层数: 6")
    print(f"  前馈网络维度: 2048")
    print(f"  Dropout: 0.1")

    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    print("\n模型结构:")
    print(model)

    print("\n子模块分解:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:30s}: {module_params:>12,} 参数 ({module_params / total_params * 100:.1f}%)")


def show_gpt_architecture():
    """展示 GPT 模型结构"""
    print("\n" + "="*80)
    print("GPT (Decoder-only) 架构")
    print("="*80)

    model = GPT(
        vocab_size=1000,
        num_layers=12,
        d_model=768,
        nhead=12,
        d_ff=3072,
        dropout=0.1,
        activation='gelu'
    )

    total_params, trainable_params = count_parameters(model)

    print("\n模型配置:")
    print(f"  词表大小: 1000")
    print(f"  层数: 12")
    print(f"  隐藏维度: 768")
    print(f"  注意力头数: 12")
    print(f"  前馈网络维度: 3072")
    print(f"  激活函数: GELU")
    print(f"  Dropout: 0.1")

    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    print("\n模型结构:")
    print(model)

    print("\n子模块分解:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:30s}: {module_params:>12,} 参数 ({module_params / total_params * 100:.1f}%)")


def compare_architectures():
    """对比不同架构"""
    print("\n" + "="*80)
    print("架构对比")
    print("="*80)

    configs = [
        {
            'name': 'Transformer (Tiny)',
            'model': Transformer(
                vocab_size=1000,
                d_model=128,
                nhead=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                d_ff=512,
                dropout=0.1
            )
        },
        {
            'name': 'Transformer (Base)',
            'model': Transformer(
                vocab_size=1000,
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                d_ff=2048,
                dropout=0.1
            )
        },
        {
            'name': 'GPT (Small)',
            'model': GPT(
                vocab_size=1000,
                num_layers=6,
                d_model=512,
                nhead=8,
                d_ff=2048,
                dropout=0.1
            )
        },
        {
            'name': 'GPT (Medium)',
            'model': GPT(
                vocab_size=1000,
                num_layers=12,
                d_model=768,
                nhead=12,
                d_ff=3072,
                dropout=0.1
            )
        }
    ]

    print(f"\n{'模型':<30s} {'参数量':<15s} {'层数':<10s} {'d_model':<10s} {'nhead':<10s}")
    print("-" * 80)

    for config in configs:
        model = config['model']
        total_params, _ = count_parameters(model)

        # 获取配置信息
        if isinstance(model, Transformer):
            num_layers = f"{model.encoder.layers.__len__()}+{model.decoder.layers.__len__()}"
            nhead = str(model.encoder.layers[0].self_attn.num_heads)
        else:
            num_layers = str(len(model.layers))
            nhead = str(model.layers[0].self_attn.num_heads)

        d_model = str(model.d_model)

        print(f"{config['name']:<30s} {total_params:>12,} ({total_params/1e6:.1f}M)  {num_layers:<10s} {d_model:<10s} {nhead:<10s}")


def show_layer_details():
    """展示单个层的详细信息"""
    print("\n" + "="*80)
    print("Transformer 层详细信息")
    print("="*80)

    model = Transformer(
        vocab_size=1000,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=2048,
        dropout=0.1
    )

    # 分析编码器层
    print("\n编码器层 (TransformerEncoderLayer):")
    encoder_layer = model.encoder.layers[0]
    for name, module in encoder_layer.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s}: {module_params:>10,} 参数")

    # 分析解码器层
    print("\n解码器层 (TransformerDecoderLayer):")
    decoder_layer = model.decoder.layers[0]
    for name, module in decoder_layer.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s}: {module_params:>10,} 参数")


if __name__ == '__main__':
    show_transformer_architecture()
    show_gpt_architecture()
    compare_architectures()
    show_layer_details()

    print("\n" + "="*80)
    print("展示完成！")
    print("="*80)
