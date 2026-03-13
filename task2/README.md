# Task2: 基于深度学习的文本分类

## 项目简介

本项目实现了基于深度学习的影评文本分类系统，支持多种模型架构（CNN、RNN、LSTM、GRU、Transformer），并提供完整的实验和可视化工具。

## 环境要求

### 必需依赖
```bash
# 核心依赖
pip install torch numpy matplotlib seaborn

# 或使用uv
uv pip install torch numpy matplotlib seaborn
```

### 可选依赖
```bash
# 用于下载GloVe词向量（如果需要使用预训练词向量）
pip install requests tqdm
```

### GPU支持
- 推荐使用CUDA支持的GPU以加速训练
- 自动检测并使用可用的GPU

## 快速开始

### 1. 数据预处理

```bash
python data_process.py
```

这将会：
- 读取`raw_data/`中的训练和测试数据
- 构建词表（8,144个词）
- 将文本转换为固定长度的索引序列（100）
- 划分训练集、验证集、测试集
- 保存处理后的数据到`temp_data/`

### 2. 训练模型

#### 基础训练（CNN模型）
```bash
python train.py --model cnn --batch_size 128 --num_epochs 15 --lr 0.001
```

#### 训练LSTM模型
```bash
python train.py --model lstm --batch_size 64 --num_epochs 10 --lr 0.001 --hidden_dim 128
```

#### 训练双向GRU模型
```bash
python train.py --model gru --batch_size 64 --num_epochs 10 --lr 0.001 --hidden_dim 128 --bidirectional
```

#### 训练Transformer模型
```bash
python train.py --model transformer --batch_size 32 --num_epochs 10 --lr 0.0001
```

### 3. 分析结果

```bash
python analysis.py --mode analyze
```

这将会：
- 加载所有训练历史
- 生成性能比较图表
- 创建汇总报告

## 完整参数说明

### train.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--model` | str | cnn | 模型类型: cnn, rnn, lstm, gru, transformer |
| `--batch_size` | int | 64 | 批次大小 |
| `--num_epochs` | int | 20 | 训练轮数 |
| `--lr` | float | 0.001 | 学习率 |
| `--optimizer` | str | adam | 优化器: sgd, adam, adamw |
| `--embedding_dim` | int | 100 | 词向量维度 |
| `--hidden_dim` | int | 128 | 隐藏层维度 (RNN/LSTM/GRU) |
| `--num_filters` | int | 100 | CNN卷积核数量 |
| `--filter_sizes` | str | 3,4,5 | CNN卷积核大小（逗号分隔） |
| `--num_layers` | int | 1 | RNN/LSTM/GRU层数 |
| `--dropout` | float | 0.5 | Dropout比例 |
| `--num_heads` | int | 8 | Transformer注意力头数 |
| `--num_transformer_layers` | int | 2 | Transformer层数 |
| `--bidirectional` | flag | False | 是否使用双向RNN/LSTM/GRU |
| `--use_glove` | flag | False | 是否使用预训练GloVe词向量 |
| `--glove_path` | str | glove.6B.100d.txt | GloVe文件路径 |
| `--seed` | int | 42 | 随机种子 |

### 示例命令

#### 实验一：不同模型比较
```bash
# CNN
python train.py --model cnn --num_epochs 15 --batch_size 128

# LSTM
python train.py --model lstm --num_epochs 10 --hidden_dim 128

# Bi-LSTM
python train.py --model lstm --num_epochs 10 --hidden_dim 128 --bidirectional

# GRU
python train.py --model gru --num_epochs 10 --hidden_dim 128

# Transformer
python train.py --model transformer --num_epochs 10 --batch_size 32
```

#### 实验二：学习率影响
```bash
python train.py --model cnn --lr 0.0001 --num_epochs 15
python train.py --model cnn --lr 0.001 --num_epochs 15
python train.py --model cnn --lr 0.01 --num_epochs 15
```

#### 实验三：优化器比较
```bash
python train.py --model cnn --optimizer sgd --lr 0.01
python train.py --model cnn --optimizer adam --lr 0.001
python train.py --model cnn --optimizer adamw --lr 0.001
```

#### 实验四：CNN参数调优
```bash
# 不同卷积核数量
python train.py --model cnn --num_filters 50
python train.py --model cnn --num_filters 100
python train.py --model cnn --num_filters 150
python train.py --model cnn --num_filters 200

# 不同卷积核大小
python train.py --model cnn --filter_sizes 2,3,4
python train.py --model cnn --filter_sizes 3,4,5
python train.py --model cnn --filter_sizes 4,5,6
```

## 项目结构

```
task2/
├── data_process.py          # 数据预处理脚本
├── train.py                 # 模型训练脚本
├── analysis.py              # 结果分析和可视化
├── README.md                # 本文件
├── raw_data/                # 原始数据
│   ├── new_train.tsv       # 训练数据 (8,528条)
│   └── new_test.tsv        # 测试数据 (3,309条)
├── temp_data/               # 预处理后的数据
│   ├── vocab.pkl           # 词表
│   ├── train_texts.npy     # 训练集文本序列
│   ├── train_labels.npy    # 训练集标签
│   ├── val_texts.npy       # 验证集文本序列
│   ├── val_labels.npy      # 验证集标签
│   ├── test_texts.npy      # 测试集文本序列
│   ├── test_labels.npy     # 测试集标签
│   └── data_info.json      # 数据信息
├── models/                  # 训练好的模型
│   ├── *.pt                # 模型权重
│   └── *_history.json      # 训练历史
└── docs/                    # 文档
    ├── project.md          # 项目说明
    ├── record.md           # 实验记录
    ├── experiment_summary.md  # 实验汇总
    └── figures/            # 可视化图表
        ├── model_comparison.png
        ├── training_curves.png
        └── ...
```

## 模型架构

### 技术实现说明

**所有模型均使用PyTorch高级API实现，无需从底层编写算法：**

- ✅ `nn.Embedding` - 词嵌入层
- ✅ `nn.Conv1d` / `nn.MaxPool1d` - 卷积和池化层
- ✅ `nn.RNN` / `nn.LSTM` / `nn.GRU` - 循环神经网络层
- ✅ `nn.TransformerEncoder` - Transformer编码器
- ✅ `nn.Linear` - 全连接层
- ✅ `nn.Dropout` / `nn.ReLU` - 正则化和激活函数

**无需手动实现任何算法的底层逻辑**，直接调用PyTorch提供的模块即可。

### CNN文本分类模型
```
Input → Embedding → Conv1d (多尺度) → MaxPool → Concat → Dropout → FC → Softmax
```

**特点**:
- 使用多个不同大小的卷积核捕捉不同尺度的特征
- 最大池化提取最重要的特征
- 适合短文本分类

### RNN模型
```
Input → Embedding → RNN → Last Hidden → Dropout → FC → Softmax
```

### LSTM模型
```
Input → Embedding → LSTM → Last Hidden → Dropout → FC → Softmax
```

### GRU模型
```
Input → Embedding → GRU → Last Hidden → Dropout → FC → Softmax
```

### Transformer模型
```
Input → Embedding → Positional Encoding → Transformer Encoder → Mean Pool → FC → Softmax
```

## 实验结果

### 模型性能比较

| 模型 | 验证集准确率 | 测试集准确率 | 参数量 |
|-----|------------|------------|-------|
| CNN | 44.89% | 44.79% | 997K |
| LSTM | 29.96% | 30.13% | 933K |
| Bi-LSTM | 29.96% | 30.13% | 1,051K |
| Bi-GRU | 29.96% | 30.13% | 992K |

### 最佳配置

```python
{
    'model_type': 'CNN',
    'embedding_dim': 100,
    'num_filters': 150,
    'filter_sizes': [3, 4, 5],
    'dropout': 0.5,
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'batch_size': 128,
    'num_epochs': 15,
}
```

## 使用预训练词向量

### 下载GloVe词向量

```bash
# 下载GloVe 6B词向量 (100维)
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# 或使用Python下载
python -c "
import requests
url = 'http://nlp.stanford.edu/data/glove.6B.zip'
response = requests.get(url, stream=True)
with open('glove.6B.zip', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
"
```

### 使用GloVe训练

```bash
python train.py --model cnn --use_glove --glove_path glove.6B.100d.txt
```

## 常见问题

### Q1: 训练时出现CUDA内存不足
**解决方案**:
- 减小batch_size
- 减小模型维度（embedding_dim, hidden_dim, num_filters）
- 使用梯度累积

### Q2: RNN类模型训练困难
**解决方案**:
- 增加训练轮数
- 调整学习率（尝试0.0001-0.01）
- 使用梯度裁剪
- 添加层归一化

### Q3: 模型过拟合
**解决方案**:
- 增加Dropout比例
- 使用L2正则化
- 早停（Early Stopping）
- 数据增强

### Q4: 训练速度慢
**解决方案**:
- 使用GPU
- 增大batch_size
- 使用混合精度训练
- 减小模型复杂度

## 扩展功能

### 添加新的模型

在`train.py`中添加新的模型类：

```python
class YourModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, ...):
        super(YourModel, self).__init__()
        # 定义模型结构

    def forward(self, x):
        # 定义前向传播
        return logits
```

然后在`main()`函数中添加相应的创建逻辑。

### 自定义损失函数

修改`train.py`中的损失函数部分：

```python
# 默认使用交叉熵损失
criterion = nn.CrossEntropyLoss()

# 可以使用其他损失函数，如：
# criterion = nn.NLLLoss()
# criterion = nn.MultiMarginLoss()
```

### 添加学习率调度器

在训练循环中添加学习率调度：

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 在每个epoch后
scheduler.step()
```

## 参考资料

1. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification
2. Graves, A. (2013). Generating Sequences With Recurrent Neural Networks
4. Vaswani et al. (2017). Attention is All You Need

