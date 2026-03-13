# Task2 实现说明

## 🎯 核心原则

**本项目的所有深度学习模型均使用PyTorch高级API实现，无需从底层手动实现任何算法。**

## ✅ 使用的PyTorch API列表

### 1. 词嵌入层
```python
nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
```

### 2. 卷积神经网络 (CNN)
```python
# 一维卷积层
nn.Conv1d(in_channels, out_channels, kernel_size)

# 最大池化层
F.max_pool1d(input, kernel_size)

# Dropout层
nn.Dropout(p=0.5)

# 全连接层
nn.Linear(in_features, out_features)
```

### 3. 循环神经网络 (RNN)
```python
# RNN层
nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

# LSTM层
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

# GRU层
nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
```

### 4. Transformer
```python
# Transformer编码器层
nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

# Transformer编码器
nn.TransformerEncoder(encoder_layer, num_layers)
```

### 5. 激活函数和损失函数
```python
# ReLU激活函数
F.relu(input)

# 交叉熵损失函数
nn.CrossEntropyLoss()
```

### 6. 优化器
```python
# SGD优化器
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam优化器
torch.optim.Adam(model.parameters(), lr=0.001)

# AdamW优化器
torch.optim.AdamW(model.parameters(), lr=0.001)
```

## 📊 代码示例

### CNN模型实现
```python
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes,
                 num_filters=100, filter_sizes=(3, 4, 5), dropout=0.5):
        super(CNNTextClassifier, self).__init__()

        # 使用PyTorch的Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 使用PyTorch的Conv1d层
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        # 使用PyTorch的Dropout层
        self.dropout = nn.Dropout(dropout)

        # 使用PyTorch的Linear层
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]

        # 卷积和池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # 使用PyTorch的ReLU
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        # 拼接
        output = torch.cat(conv_outputs, dim=1)

        # Dropout和全连接
        output = self.dropout(output)
        logits = self.fc(output)

        return logits
```

### LSTM模型实现
```python
class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.5, bidirectional=False):
        super(LSTMTextClassifier, self).__init__()

        # 使用PyTorch的Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 使用PyTorch的LSTM层
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 使用PyTorch的Dropout层
        self.dropout = nn.Dropout(dropout)

        # 使用PyTorch的Linear层
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)

        # LSTM
        output, (hidden, cell) = self.lstm(embedded)

        # 使用最后一个时间步的输出
        output = output[:, -1, :]

        # Dropout和全连接
        output = self.dropout(output)
        logits = self.fc(output)

        return logits
```

## 🎓 学习资源

如果你想了解这些API的用法，可以参考：

1. **PyTorch官方文档**
   - [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
   - [nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
   - [nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
   - [nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
   - [nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
   - [nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

2. **本项目的实现**
   - 查看 `train.py` 文件中的完整实现
   - 所有模型都使用上述API构建

## 💡 关键要点

✅ **直接使用PyTorch的高级API** - 无需从底层实现算法
✅ **组合现有模块** - 通过组合nn.Module子类构建模型
✅ **关注模型架构** - 重点在于如何组合这些模块，而不是如何实现它们
✅ **灵活配置** - 通过参数调整模型结构和超参数

---

*实现说明 - Task2 深度学习文本分类*
