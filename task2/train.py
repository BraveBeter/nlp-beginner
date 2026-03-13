#!/usr/bin/env python3
"""
模型训练脚本 - Task2 深度学习文本分类
支持CNN、RNN、LSTM、Transformer等多种模型结构
"""

import os
import json
import pickle
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam, AdamW


# ============================================
# 自定义数据集类
# ============================================

class TextDataset(Dataset):
    """文本分类数据集"""

    def __init__(self, texts, labels):
        """
        初始化数据集

        Args:
            texts: 文本序列列表 (numpy array)
            labels: 标签列表 (numpy array)
        """
        self.texts = torch.LongTensor(texts)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# ============================================
# CNN 模型
# ============================================

class CNNTextClassifier(nn.Module):
    """CNN文本分类模型"""

    def __init__(self, vocab_size, embedding_dim, num_classes,
                 num_filters=100, filter_sizes=(3, 4, 5), dropout=0.5,
                 padding_idx=0, pretrained_embeddings=None):
        """
        初始化CNN模型

        Args:
            vocab_size: 词表大小
            embedding_dim: 词向量维度
            num_classes: 类别数
            num_filters: 每种卷积核的数量
            filter_sizes: 卷积核大小列表
            dropout: dropout比例
            padding_idx: 填充索引
            pretrained_embeddings: 预训练词向量
        """
        super(CNNTextClassifier, self).__init__()

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # 如果有预训练词向量，使用它们初始化embedding
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # 可以选择是否冻结embedding层
            # self.embedding.weight.requires_grad = False

        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_len]

        Returns:
            输出logits [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # 转换维度以适应Conv1d: [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)

        # 对每个卷积核进行卷积和池化
        conv_outputs = []
        for conv in self.convs:
            # 卷积: [batch_size, num_filters, seq_len - kernel_size + 1]
            conv_out = F.relu(conv(embedded))

            # 最大池化: [batch_size, num_filters]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        # 拼接所有卷积核的输出: [batch_size, num_filters * len(filter_sizes)]
        output = torch.cat(conv_outputs, dim=1)

        # Dropout
        output = self.dropout(output)

        # 全连接层: [batch_size, num_classes]
        logits = self.fc(output)

        return logits


# ============================================
# RNN 模型
# ============================================

class RNNTextClassifier(nn.Module):
    """RNN文本分类模型"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.5, padding_idx=0,
                 pretrained_embeddings=None):
        """
        初始化RNN模型

        Args:
            vocab_size: 词表大小
            embedding_dim: 词向量维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            num_layers: RNN层数
            dropout: dropout比例
            padding_idx: 填充索引
            pretrained_embeddings: 预训练词向量
        """
        super(RNNTextClassifier, self).__init__()

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # RNN层
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_len]

        Returns:
            输出logits [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # RNN: output [batch_size, seq_len, hidden_dim]
        #      hidden [num_layers, batch_size, hidden_dim]
        output, hidden = self.rnn(embedded)

        # 使用最后一个时间步的输出
        # output[:, -1, :]: [batch_size, hidden_dim]
        output = output[:, -1, :]

        # Dropout
        output = self.dropout(output)

        # 全连接层
        logits = self.fc(output)

        return logits


# ============================================
# LSTM 模型
# ============================================

class LSTMTextClassifier(nn.Module):
    """LSTM文本分类模型"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.5, padding_idx=0,
                 pretrained_embeddings=None, bidirectional=False):
        """
        初始化LSTM模型

        Args:
            vocab_size: 词表大小
            embedding_dim: 词向量维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            num_layers: LSTM层数
            dropout: dropout比例
            padding_idx: 填充索引
            pretrained_embeddings: 预训练词向量
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMTextClassifier, self).__init__()

        self.bidirectional = bidirectional

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=bidirectional)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_len]

        Returns:
            输出logits [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # LSTM: output [batch_size, seq_len, hidden_dim * num_directions]
        #       hidden [num_layers * num_directions, batch_size, hidden_dim]
        output, (hidden, cell) = self.lstm(embedded)

        # 使用最后一个时间步的输出
        # output[:, -1, :]: [batch_size, hidden_dim * num_directions]
        output = output[:, -1, :]

        # Dropout
        output = self.dropout(output)

        # 全连接层
        logits = self.fc(output)

        return logits


# ============================================
# GRU 模型
# ============================================

class GRUTextClassifier(nn.Module):
    """GRU文本分类模型"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.5, padding_idx=0,
                 pretrained_embeddings=None, bidirectional=False):
        """
        初始化GRU模型

        Args:
            vocab_size: 词表大小
            embedding_dim: 词向量维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            num_layers: GRU层数
            dropout: dropout比例
            padding_idx: 填充索引
            pretrained_embeddings: 预训练词向量
            bidirectional: 是否使用双向GRU
        """
        super(GRUTextClassifier, self).__init__()

        self.bidirectional = bidirectional

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # GRU层
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         bidirectional=bidirectional)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_len]

        Returns:
            输出logits [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # GRU
        output, hidden = self.gru(embedded)

        # 使用最后一个时间步的输出
        output = output[:, -1, :]

        # Dropout
        output = self.dropout(output)

        # 全连接层
        logits = self.fc(output)

        return logits


# ============================================
# Transformer 模型
# ============================================

class TransformerTextClassifier(nn.Module):
    """Transformer文本分类模型"""

    def __init__(self, vocab_size, embedding_dim, num_classes,
                 num_heads=8, num_layers=6, dim_feedforward=2048,
                 dropout=0.1, padding_idx=0, max_seq_len=100,
                 pretrained_embeddings=None):
        """
        初始化Transformer模型

        Args:
            vocab_size: 词表大小
            embedding_dim: 词向量维度
            num_classes: 类别数
            num_heads: 多头注意力的头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: dropout比例
            padding_idx: 填充索引
            max_seq_len: 最大序列长度
            pretrained_embeddings: 预训练词向量
        """
        super(TransformerTextClassifier, self).__init__()

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # 位置编码
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout, max_seq_len)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 全连接层
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_len]

        Returns:
            输出logits [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # 添加位置编码
        embedded = self.pos_encoding(embedded)

        # Transformer编码器: [batch_size, seq_len, embedding_dim]
        output = self.transformer_encoder(embedded)

        # 使用第一个token（CLS token）的输出进行分类
        # 或者使用平均池化
        output = output.mean(dim=1)  # [batch_size, embedding_dim]

        # 全连接层
        logits = self.fc(output)

        return logits


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================
# 训练和评估函数
# ============================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (texts, labels) in enumerate(dataloader):
        texts, labels = texts.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(texts)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)

            # 前向传播
            outputs = model(texts)

            # 计算损失
            loss = criterion(outputs, labels)

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train(model, train_loader, val_loader, test_loader, optimizer, criterion,
          device, num_epochs, model_save_path):
    """训练模型"""

    print(f"\n开始训练 {type(model).__name__} 模型...")
    print(f"设备: {device}")
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")

    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': []
    }

    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 测试
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        # 打印进度
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
            }, model_save_path)
            print(f"  -> 保存最佳模型 (验证集准确率: {val_acc:.2f}%)")

    print(f"\n训练完成! 最佳验证集准确率: {best_val_acc:.2f}%")

    return history


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description='训练深度学习文本分类模型')
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'rnn', 'lstm', 'gru', 'transformer'],
                       help='模型类型')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam', 'adamw'], help='优化器')
    parser.add_argument('--embedding_dim', type=int, default=100, help='词向量维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_filters', type=int, default=100, help='CNN卷积核数量')
    parser.add_argument('--filter_sizes', type=str, default='3,4,5', help='CNN卷积核大小')
    parser.add_argument('--num_layers', type=int, default=1, help='RNN/LSTM/GRU层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
    parser.add_argument('--num_heads', type=int, default=8, help='Transformer注意力头数')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--bidirectional', action='store_true', help='是否使用双向RNN/LSTM/GRU')
    parser.add_argument('--use_glove', action='store_true', help='是否使用预训练GloVe词向量')
    parser.add_argument('--glove_path', type=str, default='glove.6B.100d.txt',
                       help='GloVe词向量文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_data_dir = os.path.join(base_dir, 'temp_data')
    models_dir = os.path.join(base_dir, 'models')

    os.makedirs(models_dir, exist_ok=True)

    print("\n加载数据...")
    train_texts = np.load(os.path.join(temp_data_dir, 'train_texts.npy'))
    train_labels = np.load(os.path.join(temp_data_dir, 'train_labels.npy'))
    val_texts = np.load(os.path.join(temp_data_dir, 'val_texts.npy'))
    val_labels = np.load(os.path.join(temp_data_dir, 'val_labels.npy'))
    test_texts = np.load(os.path.join(temp_data_dir, 'test_texts.npy'))
    test_labels = np.load(os.path.join(temp_data_dir, 'test_labels.npy'))

    # 加载数据信息
    with open(os.path.join(temp_data_dir, 'data_info.json'), 'r') as f:
        data_info = json.load(f)

    vocab_size = data_info['vocab_size']
    num_classes = data_info['num_classes']
    max_seq_len = data_info['max_seq_len']
    padding_idx = data_info['padding_idx']

    print(f"词表大小: {vocab_size}")
    print(f"类别数: {num_classes}")
    print(f"序列长度: {max_seq_len}")

    # 加载GloVe词向量（如果指定）
    pretrained_embeddings = None
    if args.use_glove:
        print(f"\n加载GloVe词向量: {args.glove_path}")
        # 这里需要实现GloVe加载逻辑
        # 为了简化，暂时跳过
        print("警告: GloVe加载功能待实现")

    # 创建数据加载器
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)
    test_dataset = TextDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    print(f"\n创建 {args.model.upper()} 模型...")

    if args.model == 'cnn':
        filter_sizes = tuple(map(int, args.filter_sizes.split(',')))
        model = CNNTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            num_classes=num_classes,
            num_filters=args.num_filters,
            filter_sizes=filter_sizes,
            dropout=args.dropout,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings
        )
    elif args.model == 'rnn':
        model = RNNTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings
        )
    elif args.model == 'lstm':
        model = LSTMTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings,
            bidirectional=args.bidirectional
        )
    elif args.model == 'gru':
        model = GRUTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            padding_idx=padding_idx,
            pretrained_embeddings=pretrained_embeddings,
            bidirectional=args.bidirectional
        )
    elif args.model == 'transformer':
        model = TransformerTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            num_classes=num_classes,
            num_heads=args.num_heads,
            num_layers=args.num_transformer_layers,
            dropout=args.dropout,
            padding_idx=padding_idx,
            max_seq_len=max_seq_len,
            pretrained_embeddings=pretrained_embeddings
        )

    model = model.to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 创建优化器
    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr)

    # 创建损失函数
    criterion = nn.CrossEntropyLoss()

    # 生成模型保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_{timestamp}"
    model_save_path = os.path.join(models_dir, f"{model_name}.pt")

    # 训练模型
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        model_save_path=model_save_path
    )

    # 保存训练历史
    history_path = os.path.join(models_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n训练历史已保存到: {history_path}")
    print(f"模型已保存到: {model_save_path}")


if __name__ == '__main__':
    main()
