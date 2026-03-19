# Transformer 从零实现

基于《Attention Is All You Need》论文，从零实现完整的 Transformer 架构（仅使用 PyTorch 基础组件）。

## 项目特点

- ✅ **完全从零实现**：不依赖 `nn.Transformer`，手动实现所有核心组件
- ✅ **高度模块化**：清晰的模块划分，便于理解和扩展
- ✅ **Type Hint 注解**：完整的类型提示
- ✅ **详细的中文注释**：每个模块都有详细的说明
- ✅ **双架构支持**：Encoder-Decoder 和 Decoder-only (GPT)
- ✅ **完整测试**：包含所有组件的单元测试

## 目录结构

```
task3/
├── models/                          # 核心架构实现
│   ├── __init__.py
│   ├── attention.py                 # 多头注意力机制
│   ├── encoder.py                   # Encoder 层和堆叠
│   ├── decoder.py                   # Decoder 层和堆叠
│   ├── transformer.py               # 完整 Transformer (Enc-Dec)
│   ├── gpt.py                       # Decoder-only (GPT)
│   └── modules/                     # 基础组件
│       ├── __init__.py
│       ├── positional_encoding.py   # 位置编码
│       ├── feed_forward.py          # 前馈网络
│       └── norm.py                  # 层归一化
├── test_models.py                   # 模型测试脚本
└── README.md                        # 项目说明
```

## 核心组件

### 1. 基础组件 (modules/)

#### PositionalEncoding
- 正弦/余弦位置编码
- 支持变长序列的外推

#### LayerNorm
- 沿特征维度归一化
- 适合序列建模任务

#### FeedForward
- 两层全连接网络
- 支持 ReLU/GELU 激活函数

### 2. 注意力机制 (attention.py)

#### MultiheadAttention
- 缩放点积注意力
- 支持填充掩码和因果掩码
- 支持自注意力和编码器-解码器注意力

### 3. 编码器 (encoder.py)

#### TransformerEncoderLayer
- 多头自注意力
- 前馈网络
- 残差连接 + 层归一化

#### TransformerEncoder
- 堆叠多个编码器层
- 添加位置编码

### 4. 解码器 (decoder.py)

#### TransformerDecoderLayer
- Masked 多头自注意力
- 编码器-解码器注意力
- 前馈网络
- 残差连接 + 层归一化

#### TransformerDecoder
- 堆叠多个解码器层

### 5. 完整模型

#### Transformer (transformer.py)
- 标准 Encoder-Decoder 架构
- 支持训练模式（Teacher Forcing）
- 支持推理模式（自回归生成）

#### GPT (gpt.py)
- Decoder-only 架构
- 支持多种采样策略（贪心、top-k、top-p）
- 适合语言建模和文本生成

## 测试

运行测试脚本验证所有组件：

```bash
cd task3
python test_models.py
```

测试覆盖：
- ✅ 位置编码
- ✅ 层归一化
- ✅ 前馈网络
- ✅ 多头注意力（无掩码、填充掩码、因果掩码）
- ✅ 编码器层
- ✅ 解码器层
- ✅ 完整 Transformer（训练和推理）
- ✅ GPT 模型（前向传播和生成）

## 快速开始

### 1. 使用 Transformer (Encoder-Decoder)

```python
from models import Transformer

# 创建模型
model = Transformer(
    vocab_size=1000,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    dropout=0.1
)

# 训练模式
src = torch.randint(0, 1000, (32, 50))  # [batch_size, src_len]
tgt = torch.randint(0, 1000, (32, 30))  # [batch_size, tgt_len]
tgt_mask = model.generate_square_subsequent_mask(30)

output = model(src, tgt, tgt_mask=tgt_mask)  # [32, 30, 1000]

# 推理模式
generated = model.generate(src, max_len=50, bos_idx=1, eos_idx=2)
```

### 2. 使用 GPT (Decoder-only)

```python
from models import GPT

# 创建模型
model = GPT(
    vocab_size=1000,
    num_layers=12,
    d_model=768,
    nhead=12,
    d_ff=3072,
    dropout=0.1
)

# 训练模式
x = torch.randint(0, 1000, (32, 128))  # [batch_size, seq_len]
output = model(x)  # [32, 128, 1000]

# 推理模式
prompt = torch.randint(0, 1000, (32, 10))
generated = model.generate(
    prompt,
    max_len=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

## 架构对比

### Encoder-Decoder (Transformer)
- **适用场景**：序列到序列任务（翻译、摘要等）
- **优势**：Encoder 充分理解源序列，Decoder 逐步生成
- **结构**：Encoder + Decoder（包含 Cross-Attention）

### Decoder-only (GPT)
- **适用场景**：生成任务（续写、对话等）
- **优势**：结构简单，训练稳定，适合大规模预训练
- **结构**：仅 Decoder（只有 Self-Attention）

## 技术亮点

1. **手动实现注意力机制**：
   - 缩放点积注意力
   - 多头注意力
   - 掩码机制

2. **灵活的掩码支持**：
   - Padding mask：屏蔽填充 token
   - Causal mask：防止信息泄露（自回归）

3. **残差连接 + 层归一化**：
   - Post-LN（原论文）：Sublayer(x) + Norm(x)
   - Pre-LN（现代）：Norm(x) + Sublayer(x)

4. **Xavier 初始化**：
   - 与原论文一致
   - 稳定训练

## 模型参数示例

| 模型 | 层数 | d_model | nhead | d_ff | 参数量 |
|------|------|---------|-------|------|--------|
| Transformer (base) | 6+6 | 512 | 8 | 2048 | ~15.7M |
| GPT (small) | 12 | 768 | 12 | 3072 | ~13.6M |

## 下一步

- [ ] 实现加法任务的数据生成和训练
- [ ] 实现语言模型的数据处理和训练
- [ ] 添加训练和评估脚本
- [ ] 实现可视化工具（Attention Map、Loss 曲线）
- [ ] 添加消融实验和超参数调优

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Harvard CS224n Transformer Tutorial](https://cs224n.stanford.edu/)
