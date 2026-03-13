#!/usr/bin/env python3
"""
数据预处理脚本 - Task2 深度学习文本分类
处理raw_data中的数据，生成模型训练所需的序列数据
"""

import os
import re
import pickle
import json
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self, max_seq_len: int = 100, min_freq: int = 2):
        """
        初始化预处理器

        Args:
            max_seq_len: 序列最大长度
            min_freq: 最小词频阈值
        """
        self.max_seq_len = max_seq_len
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def clean_text(self, text: str) -> str:
        """
        清洗文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # 转换为小写
        text = text.lower()

        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)

        # 移除URL
        text = re.sub(r'http\S+|www\S+', '', text)

        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)

        # 移除数字（保留）
        # text = re.sub(r'\d+', '', text)

        # 只保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        分词

        Args:
            text: 清洗后的文本

        Returns:
            词语列表
        """
        # 简单的空格分词
        tokens = text.split()
        return tokens

    def build_vocab(self, texts: List[str]) -> None:
        """
        构建词表

        Args:
            texts: 文本列表
        """
        # 统计词频
        word_freq = Counter()

        for text in texts:
            cleaned = self.clean_text(text)
            tokens = self.tokenize(cleaned)
            word_freq.update(tokens)

        # 过滤低频词并构建词表
        # 保留特殊标记
        self.word2idx = {
            '<PAD>': 0,  # 填充标记
            '<UNK>': 1,  # 未知词标记
        }

        # 添加高频词
        for word, freq in word_freq.most_common():
            if freq >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        print(f"词表构建完成，共 {self.vocab_size} 个词")

    def text_to_sequence(self, text: str) -> List[int]:
        """
        将文本转换为索引序列

        Args:
            text: 原始文本

        Returns:
            索引序列
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)

        # 转换为索引
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

        # 截断或填充到固定长度
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        else:
            sequence = sequence + [self.word2idx['<PAD>']] * (self.max_seq_len - len(sequence))

        return sequence

    def save_vocab(self, save_path: str) -> None:
        """
        保存词表

        Args:
            save_path: 保存路径
        """
        with open(save_path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size,
                'max_seq_len': self.max_seq_len,
            }, f)

        print(f"词表已保存到 {save_path}")

    def load_vocab(self, load_path: str) -> None:
        """
        加载词表

        Args:
            load_path: 加载路径
        """
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.vocab_size = data['vocab_size']
            self.max_seq_len = data['max_seq_len']

        print(f"词表已加载，共 {self.vocab_size} 个词")


def read_tsv(file_path: str) -> Tuple[List[str], List[int]]:
    """
    读取TSV文件

    Args:
        file_path: 文件路径

    Returns:
        (文本列表, 标签列表)
    """
    texts = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过可能的header
        lines = f.readlines()

        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # 最后一列是标签
                label = int(parts[-1])
                # 其余部分是文本（可能包含tab）
                text = '\t'.join(parts[:-1])
                texts.append(text)
                labels.append(label)

    return texts, labels


def split_train_val(texts: List[int], labels: List[int], val_ratio: float = 0.1) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    划分训练集和验证集（分层采样）

    Args:
        texts: 文本序列列表
        labels: 标签列表
        val_ratio: 验证集比例

    Returns:
        (train_texts, val_texts, train_labels, val_labels)
    """
    import random

    # 按标签分组
    label_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    # 对每个类别进行分层采样
    val_indices = []
    train_indices = []

    for label, indices in label_to_indices.items():
        # 随机打乱
        random.seed(42)
        random.shuffle(indices)

        # 计算验证集大小
        val_size = int(len(indices) * val_ratio)

        # 划分
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])

    # 根据索引划分数据
    val_texts_final = [texts[i] for i in val_indices]
    val_labels_final = [labels[i] for i in val_indices]
    train_texts_final = [texts[i] for i in train_indices]
    train_labels_final = [labels[i] for i in train_indices]

    return train_texts_final, val_texts_final, train_labels_final, val_labels_final


def main():
    """主函数"""

    # 路径配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, 'raw_data')
    temp_data_dir = os.path.join(base_dir, 'temp_data')

    # 确保目录存在
    os.makedirs(temp_data_dir, exist_ok=True)

    print("=" * 50)
    print("Task2 数据预处理开始")
    print("=" * 50)

    # 初始化预处理器
    preprocessor = TextPreprocessor(max_seq_len=100, min_freq=2)

    # 读取训练数据
    print("\n[1/5] 读取训练数据...")
    train_texts, train_labels = read_tsv(os.path.join(raw_data_dir, 'new_train.tsv'))
    print(f"训练数据: {len(train_texts)} 条")

    # 构建词表
    print("\n[2/5] 构建词表...")
    preprocessor.build_vocab(train_texts)

    # 保存词表
    vocab_path = os.path.join(temp_data_dir, 'vocab.pkl')
    preprocessor.save_vocab(vocab_path)

    # 转换训练数据
    print("\n[3/5] 转换训练数据为序列...")
    train_sequences = [preprocessor.text_to_sequence(text) for text in train_texts]
    print(f"训练序列: {len(train_sequences)} 条, 序列长度: {len(train_sequences[0])}")

    # 划分训练集和验证集
    print("\n[4/5] 划分训练集和验证集...")
    train_texts_split, val_texts, train_labels_split, val_labels = split_train_val(
        train_sequences, train_labels, val_ratio=0.1
    )
    print(f"训练集: {len(train_texts_split)} 条")
    print(f"验证集: {len(val_texts)} 条")

    # 读取并转换测试数据
    print("\n[5/5] 读取并转换测试数据...")
    test_texts_raw, test_labels = read_tsv(os.path.join(raw_data_dir, 'new_test.tsv'))
    test_sequences = [preprocessor.text_to_sequence(text) for text in test_texts_raw]
    print(f"测试数据: {len(test_sequences)} 条")

    # 保存处理后的数据
    print("\n保存处理后的数据...")

    # 转换为numpy数组并保存
    np.save(os.path.join(temp_data_dir, 'train_texts.npy'), np.array(train_texts_split))
    np.save(os.path.join(temp_data_dir, 'train_labels.npy'), np.array(train_labels_split))
    np.save(os.path.join(temp_data_dir, 'val_texts.npy'), np.array(val_texts))
    np.save(os.path.join(temp_data_dir, 'val_labels.npy'), np.array(val_labels))
    np.save(os.path.join(temp_data_dir, 'test_texts.npy'), np.array(test_sequences))
    np.save(os.path.join(temp_data_dir, 'test_labels.npy'), np.array(test_labels))

    # 保存数据信息
    data_info = {
        'vocab_size': preprocessor.vocab_size,
        'max_seq_len': preprocessor.max_seq_len,
        'num_classes': 5,
        'train_size': len(train_texts_split),
        'val_size': len(val_texts),
        'test_size': len(test_sequences),
        'padding_idx': preprocessor.word2idx['<PAD>'],
        'unk_idx': preprocessor.word2idx['<UNK>'],
    }

    with open(os.path.join(temp_data_dir, 'data_info.json'), 'w') as f:
        json.dump(data_info, f, indent=2)

    print(f"数据信息已保存到 data_info.json")

    print("\n" + "=" * 50)
    print("数据预处理完成！")
    print("=" * 50)
    print(f"\n数据统计:")
    print(f"  词表大小: {data_info['vocab_size']}")
    print(f"  序列长度: {data_info['max_seq_len']}")
    print(f"  类别数: {data_info['num_classes']}")
    print(f"  训练集: {data_info['train_size']} 条")
    print(f"  验证集: {data_info['val_size']} 条")
    print(f"  测试集: {data_info['test_size']} 条")
    print(f"\n所有数据已保存到 {temp_data_dir}/")


if __name__ == '__main__':
    main()
