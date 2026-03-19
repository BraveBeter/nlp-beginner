"""
Transformer 模型测试脚本

测试所有核心组件的功能，确保代码正确性。
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.modules import PositionalEncoding, FeedForward, LayerNorm
from models.attention import MultiheadAttention
from models.encoder import TransformerEncoderLayer, TransformerEncoder
from models.decoder import TransformerDecoderLayer, TransformerDecoder
from models.transformer import Transformer
from models.gpt import GPT


def test_positional_encoding():
    """测试位置编码"""
    print("\n" + "="*60)
    print("测试 1: PositionalEncoding")
    print("="*60)

    d_model = 512
    max_len = 100
    batch_size = 4
    seq_len = 50

    pos_encoding = PositionalEncoding(d_model, dropout=0.0, max_len=max_len)

    # 测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入 shape: {x.shape}")

    # 前向传播
    output = pos_encoding(x)
    print(f"输出 shape: {output.shape}")

    # 检查位置编码是否正确加到输入上
    assert output.shape == x.shape, "输出形状不匹配"
    assert torch.allclose(output[:, :10, :10], x[:, :10, :10] + pos_encoding.pe[:, :10, :10], atol=1e-6), "位置编码计算错误"

    print("✓ PositionalEncoding 测试通过")


def test_layer_norm():
    """测试层归一化"""
    print("\n" + "="*60)
    print("测试 2: LayerNorm")
    print("="*60)

    d_model = 512
    batch_size = 4
    seq_len = 50

    layer_norm = LayerNorm(d_model)

    # 测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入 shape: {x.shape}")

    # 前向传播
    output = layer_norm(x)
    print(f"输出 shape: {output.shape}")

    # 检查归一化是否正确（均值接近0，方差接近1）
    mean = output.mean(dim=-1)
    var = output.var(dim=-1)

    print(f"归一化后的均值（应该接近0）: {mean.mean().item():.6f}")
    print(f"归一化后的方差（应该接近1）: {var.mean().item():.6f}")

    assert output.shape == x.shape, "输出形状不匹配"
    assert abs(mean.mean().item()) < 1e-5, "归一化后的均值不为0"
    assert abs(var.mean().item() - 1.0) < 1e-5, "归一化后的方差不为1"

    print("✓ LayerNorm 测试通过")


def test_feed_forward():
    """测试前馈网络"""
    print("\n" + "="*60)
    print("测试 3: FeedForward")
    print("="*60)

    d_model = 512
    d_ff = 2048
    batch_size = 4
    seq_len = 50

    ffn = FeedForward(d_model, d_ff, dropout=0.0)

    # 测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入 shape: {x.shape}")

    # 前向传播
    output = ffn(x)
    print(f"输出 shape: {output.shape}")

    assert output.shape == x.shape, "输出形状不匹配"

    print("✓ FeedForward 测试通过")


def test_multihead_attention():
    """测试多头注意力"""
    print("\n" + "="*60)
    print("测试 4: MultiheadAttention")
    print("="*60)

    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_len_q = 20
    seq_len_k = 30

    attn = MultiheadAttention(d_model, num_heads, dropout=0.0)

    # 测试输入
    query = torch.randn(batch_size, seq_len_q, d_model)
    key = torch.randn(batch_size, seq_len_k, d_model)
    value = torch.randn(batch_size, seq_len_k, d_model)

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")

    # 测试 1: 无掩码
    print("\n  测试 4.1: 无掩码")
    output, attn_weights = attn(query, key, value, need_weights=True)
    print(f"  输出 shape: {output.shape}")
    print(f"  注意力权重 shape: {attn_weights.shape}")

    assert output.shape == (batch_size, seq_len_q, d_model), "输出形状不匹配"
    assert attn_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k), "注意力权重形状不匹配"

    # 测试 2: 带填充掩码
    print("\n  测试 4.2: 带填充掩码")
    key_padding_mask = torch.zeros(batch_size, seq_len_k, dtype=torch.bool)
    key_padding_mask[:, 20:] = True  # 屏蔽最后10个位置

    output, attn_weights = attn(query, key, value, key_padding_mask=key_padding_mask, need_weights=True)
    print(f"  输出 shape: {output.shape}")
    print(f"  注意力权重 shape: {attn_weights.shape}")

    # 检查被屏蔽的位置的注意力权重是否为0
    assert output.shape == (batch_size, seq_len_q, d_model), "输出形状不匹配"
    assert torch.all(attn_weights[:, :, :, 20:] == 0), "被屏蔽的位置的注意力权重不为0"

    # 测试 3: 带因果掩码
    print("\n  测试 4.3: 带因果掩码")
    attn_mask = torch.triu(torch.ones(seq_len_q, seq_len_q, dtype=torch.bool), diagonal=1)

    output, attn_weights = attn(query, query, query, attn_mask=attn_mask, need_weights=True)
    print(f"  输出 shape: {output.shape}")
    print(f"  注意力权重 shape: {attn_weights.shape}")

    # 检查下三角矩阵的注意力权重是否合理
    assert output.shape == (batch_size, seq_len_q, d_model), "输出形状不匹配"
    assert torch.all(attn_weights.masked_select(attn_mask.unsqueeze(0).unsqueeze(0)) == 0), "被屏蔽的位置的注意力权重不为0"

    print("✓ MultiheadAttention 测试通过")


def test_encoder_layer():
    """测试编码器层"""
    print("\n" + "="*60)
    print("测试 5: TransformerEncoderLayer")
    print("="*60)

    d_model = 512
    nhead = 8
    d_ff = 2048
    batch_size = 4
    seq_len = 50

    encoder_layer = TransformerEncoderLayer(d_model, nhead, d_ff, dropout=0.0)

    # 测试输入
    src = torch.randn(batch_size, seq_len, d_model)
    print(f"输入 shape: {src.shape}")

    # 测试 1: 无掩码
    print("\n  测试 5.1: 无掩码")
    output = encoder_layer(src)
    print(f"  输出 shape: {output.shape}")

    assert output.shape == src.shape, "输出形状不匹配"

    # 测试 2: 带填充掩码
    print("\n  测试 5.2: 带填充掩码")
    src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    src_key_padding_mask[:, 40:] = True  # 屏蔽最后10个位置

    output = encoder_layer(src, src_key_padding_mask=src_key_padding_mask)
    print(f"  输出 shape: {output.shape}")

    assert output.shape == src.shape, "输出形状不匹配"

    print("✓ TransformerEncoderLayer 测试通过")


def test_decoder_layer():
    """测试解码器层"""
    print("\n" + "="*60)
    print("测试 6: TransformerDecoderLayer")
    print("="*60)

    d_model = 512
    nhead = 8
    d_ff = 2048
    batch_size = 4
    src_len = 50
    tgt_len = 30

    decoder_layer = TransformerDecoderLayer(d_model, nhead, d_ff, dropout=0.0)

    # 测试输入
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)

    print(f"目标序列 shape: {tgt.shape}")
    print(f"编码器输出 shape: {memory.shape}")

    # 测试 1: 无掩码
    print("\n  测试 6.1: 无掩码")
    output = decoder_layer(tgt, memory)
    print(f"  输出 shape: {output.shape}")

    assert output.shape == tgt.shape, "输出形状不匹配"

    # 测试 2: 带因果掩码
    print("\n  测试 6.2: 带因果掩码")
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool), diagonal=1)

    output = decoder_layer(tgt, memory, tgt_mask=tgt_mask)
    print(f"  输出 shape: {output.shape}")

    assert output.shape == tgt.shape, "输出形状不匹配"

    print("✓ TransformerDecoderLayer 测试通过")


def test_transformer():
    """测试完整的 Transformer 模型"""
    print("\n" + "="*60)
    print("测试 7: Transformer (Encoder-Decoder)")
    print("="*60)

    vocab_size = 1000
    d_model = 512
    nhead = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    batch_size = 4
    src_len = 50
    tgt_len = 30

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=2048,
        dropout=0.0
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试输入
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    print(f"源序列 shape: {src.shape}")
    print(f"目标序列 shape: {tgt.shape}")

    # 测试 1: 前向传播（训练模式）
    print("\n  测试 7.1: 前向传播（训练模式）")
    tgt_mask = model.generate_square_subsequent_mask(tgt_len)

    output = model(src, tgt, tgt_mask=tgt_mask)
    print(f"  输出 logits shape: {output.shape}")

    assert output.shape == (batch_size, tgt_len, vocab_size), "输出形状不匹配"

    # 测试 2: 编码器
    print("\n  测试 7.2: 编码器")
    memory = model.encode(src)
    print(f"  编码器输出 shape: {memory.shape}")

    assert memory.shape == (batch_size, src_len, d_model), "编码器输出形状不匹配"

    # 测试 3: 解码器
    print("\n  测试 7.3: 解码器")
    output = model.decode(tgt, memory, tgt_mask=tgt_mask)
    print(f"  解码器输出 shape: {output.shape}")

    assert output.shape == (batch_size, tgt_len, d_model), "解码器输出形状不匹配"

    # 测试 4: 生成（推理模式）
    print("\n  测试 7.4: 自回归生成")
    bos_idx = 1
    eos_idx = 2
    max_len = 20

    generated = model.generate(src, max_len, bos_idx, eos_idx, temperature=0)
    print(f"  生成序列 shape: {generated.shape}")
    print(f"  生成序列（第一个样本）: {generated[0].cpu().numpy()}")

    assert generated.shape[0] == batch_size, "生成序列的 batch_size 不匹配"
    assert generated.shape[1] <= max_len, "生成序列长度超过最大长度"

    print("✓ Transformer 测试通过")


def test_gpt():
    """测试 GPT 模型"""
    print("\n" + "="*60)
    print("测试 8: GPT (Decoder-only)")
    print("="*60)

    vocab_size = 1000
    num_layers = 4
    d_model = 512
    nhead = 8
    batch_size = 4
    seq_len = 50

    model = GPT(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        nhead=nhead,
        d_ff=2048,
        dropout=0.0
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试输入
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入 shape: {x.shape}")

    # 测试 1: 前向传播
    print("\n  测试 8.1: 前向传播")
    output = model(x)
    print(f"  输出 logits shape: {output.shape}")

    assert output.shape == (batch_size, seq_len, vocab_size), "输出形状不匹配"

    # 测试 2: 生成
    print("\n  测试 8.2: 自回归生成")
    prompt_len = 10
    max_len = 30
    prompt = x[:, :prompt_len]

    generated = model.generate(prompt, max_len, temperature=0.7, top_k=50, top_p=0.9)
    print(f"  生成序列 shape: {generated.shape}")
    print(f"  提示词长度: {prompt_len}")
    print(f"  生成序列长度: {generated.shape[1]}")
    print(f"  生成的 token 数量: {generated.shape[1] - prompt_len}")

    assert generated.shape[0] == batch_size, "生成序列的 batch_size 不匹配"
    assert generated.shape[1] <= max_len, "生成序列长度超过最大长度"
    assert torch.all(generated[:, :prompt_len] == prompt), "提示词被修改"

    print("✓ GPT 测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始运行 Transformer 模型测试")
    print("="*60)

    try:
        test_positional_encoding()
        test_layer_norm()
        test_feed_forward()
        test_multihead_attention()
        test_encoder_layer()
        test_decoder_layer()
        test_transformer()
        test_gpt()

        print("\n" + "="*60)
        print("✓ 所有测试通过！")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ 测试失败: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)

    # 运行所有测试
    run_all_tests()
