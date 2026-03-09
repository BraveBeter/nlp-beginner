"""
数据预处理脚本
实现 Bag of Words 和 N-gram (N<=3) 特征提取
"""

import os
import pickle
import re
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np

# 设置路径
RAW_DATA_DIR = "raw_data"
TEMP_DATA_DIR = "temp_data"


def load_data(filepath: str) -> Tuple[List[str], List[int]]:
    """
    加载 TSV 格式的数据
    返回: (文本列表, 标签列表)
    """
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 最后一列是标签，前面是文本
            parts = line.rsplit('\t', 1)
            if len(parts) == 2:
                text, label = parts
                texts.append(text)
                labels.append(int(label))
    return texts, labels


def preprocess_text(text: str) -> List[str]:
    """
    文本预处理：小写化、去除特殊字符、分词
    """
    # 转小写
    text = text.lower()
    # 只保留字母和空格
    text = re.sub(r'[^a-z\s]', '', text)
    # 分词
    tokens = text.split()
    return tokens


def build_vocabulary(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    """
    构建词汇表
    min_freq: 最小词频阈值
    """
    counter = Counter()
    for text in texts:
        tokens = preprocess_text(text)
        counter.update(tokens)

    # 过滤低频词
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    """
    提取 n-gram 特征
    """
    if n == 1:
        return tokens
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams


def build_ngram_vocabulary(texts: List[str], n: int, min_freq: int = 2) -> Dict[str, int]:
    """
    构建 n-gram 词汇表
    """
    counter = Counter()
    for text in texts:
        tokens = preprocess_text(text)
        ngrams = extract_ngrams(tokens, n)
        counter.update(ngrams)

    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for ngram, freq in counter.items():
        if freq >= min_freq:
            vocab[ngram] = idx
            idx += 1
    return vocab


def text_to_bow(text: str, vocab: Dict[str, int]) -> np.ndarray:
    """
    将文本转换为 Bag of Words 向量
    """
    tokens = preprocess_text(text)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for token in tokens:
        if token in vocab:
            vec[vocab[token]] += 1
    return vec


def text_to_ngram(text: str, vocab: Dict[str, int], n: int) -> np.ndarray:
    """
    将文本转换为 n-gram 向量
    """
    tokens = preprocess_text(text)
    ngrams = extract_ngrams(tokens, n)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for ngram in ngrams:
        if ngram in vocab:
            vec[vocab[ngram]] += 1
    return vec


def process_bag_of_words(train_texts: List[str], test_texts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    处理 Bag of Words 特征
    """
    print("Building Bag of Words vocabulary...")
    vocab = build_vocabulary(train_texts, min_freq=2)
    print(f"Vocabulary size: {len(vocab)}")

    print("Vectorizing training data...")
    X_train = np.array([text_to_bow(text, vocab) for text in train_texts])

    print("Vectorizing test data...")
    X_test = np.array([text_to_bow(text, vocab) for text in test_texts])

    return X_train, X_test, vocab


def process_ngram(train_texts: List[str], test_texts: List[str], n: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    处理 N-gram 特征
    """
    print(f"Building {n}-gram vocabulary...")
    vocab = build_ngram_vocabulary(train_texts, n, min_freq=2)
    print(f"{n}-gram vocabulary size: {len(vocab)}")

    print(f"Vectorizing training data with {n}-gram...")
    X_train = np.array([text_to_ngram(text, vocab, n) for text in train_texts])

    print(f"Vectorizing test data with {n}-gram...")
    X_test = np.array([text_to_ngram(text, vocab, n) for text in test_texts])

    return X_train, X_test, vocab


def save_processed_data(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        vocab: Dict, filename: str):
    """
    保存处理后的数据
    """
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    filepath = os.path.join(TEMP_DATA_DIR, filename)

    data = {
        'X_train': X_train,
        'y_train': np.array(y_train),
        'X_test': X_test,
        'y_test': np.array(y_test),
        'vocab': vocab
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {filepath}")


def main():
    """
    主函数：处理所有特征提取方法
    """
    print("=" * 50)
    print("Data Processing - Text Classification")
    print("=" * 50)

    # 加载原始数据
    print("\nLoading raw data...")
    train_texts, y_train = load_data(os.path.join(RAW_DATA_DIR, "new_train.tsv"))
    test_texts, y_test = load_data(os.path.join(RAW_DATA_DIR, "new_test.tsv"))

    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Number of classes: {len(set(y_train))}")

    # Bag of Words
    print("\n" + "=" * 50)
    print("Processing Bag of Words...")
    print("=" * 50)
    X_train_bow, X_test_bow, vocab_bow = process_bag_of_words(train_texts, test_texts)
    save_processed_data(X_train_bow, y_train, X_test_bow, y_test, vocab_bow, "bow_data.pkl")

    # Unigram (等同于 Bag of Words)
    print("\n" + "=" * 50)
    print("Processing Unigram...")
    print("=" * 50)
    X_train_uni, X_test_uni, vocab_uni = process_ngram(train_texts, test_texts, n=1)
    save_processed_data(X_train_uni, y_train, X_test_uni, y_test, vocab_uni, "unigram_data.pkl")

    # Bigram
    print("\n" + "=" * 50)
    print("Processing Bigram...")
    print("=" * 50)
    X_train_bi, X_test_bi, vocab_bi = process_ngram(train_texts, test_texts, n=2)
    save_processed_data(X_train_bi, y_train, X_test_bi, y_test, vocab_bi, "bigram_data.pkl")

    # Trigram
    print("\n" + "=" * 50)
    print("Processing Trigram...")
    print("=" * 50)
    X_train_tri, X_test_tri, vocab_tri = process_ngram(train_texts, test_texts, n=3)
    save_processed_data(X_train_tri, y_train, X_test_tri, y_test, vocab_tri, "trigram_data.pkl")

    print("\n" + "=" * 50)
    print("Data processing completed!")
    print("=" * 50)

    # 打印数据统计
    print("\nData Statistics:")
    print("-" * 50)
    print(f"Bag of Words: {X_train_bow.shape[1]} features")
    print(f"Unigram: {X_train_uni.shape[1]} features")
    print(f"Bigram: {X_train_bi.shape[1]} features")
    print(f"Trigram: {X_train_tri.shape[1]} features")


if __name__ == "__main__":
    main()
