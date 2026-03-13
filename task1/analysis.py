"""
结果分析与可视化脚本
对比不同方法和超参数的性能
"""

import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from collections import defaultdict

# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置路径
TEMP_DATA_DIR = "temp_data"
MODELS_DIR = "models"
DOCS_DIR = "docs"


class SoftmaxClassifier:
    """Softmax 分类器（用于加载模型）"""

    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.W = None
        self.b = None

    def forward(self, X):
        return X @ self.W + self.b

    def predict(self, X):
        logits = self.forward(X)
        return logits.argmax(dim=1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return (preds == y).float().mean().item()

    @classmethod
    def load(cls, filepath: str, learning_rate: float = 0.01):
        checkpoint = torch.load(filepath)
        model = cls(checkpoint['input_dim'], checkpoint['num_classes'], learning_rate)
        model.W = checkpoint['W']
        model.b = checkpoint['b']
        return model


class PerceptronClassifier:
    """感知机分类器（用于加载模型）"""

    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.W = None
        self.b = None

    def forward(self, X):
        return X @ self.W + self.b

    def predict(self, X):
        scores = self.forward(X)
        return scores.argmax(dim=1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return (preds == y).float().mean().item()

    @classmethod
    def load(cls, filepath: str, learning_rate: float = 0.01):
        checkpoint = torch.load(filepath)
        model = cls(checkpoint['input_dim'], checkpoint['num_classes'], learning_rate)
        model.W = checkpoint['W']
        model.b = checkpoint['b']
        return model


def load_data(filename: str):
    """加载处理后的数据"""
    import torch
    filepath = os.path.join(TEMP_DATA_DIR, filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)
    return X_test, y_test, len(data['vocab'])


def load_history(filename: str) -> Dict:
    """加载训练历史"""
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def compute_confusion_matrix(y_true, y_pred, num_classes: int = 5) -> np.ndarray:
    """计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()


def plot_training_curves(histories: Dict[str, Dict], metric: str = 'val_acc', title: str = "Training Curves"):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))

    for name, history in histories.items():
        if metric in history:
            plt.plot(history[metric], label=name, marker='o', markersize=3)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy' if metric == 'val_acc' else 'Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_comparison_bar(results: Dict[str, float], title: str = "Model Comparison"):
    """绘制对比柱状图"""
    plt.figure(figsize=(10, 6))

    names = list(results.keys())
    values = list(results.values())

    bars = plt.bar(range(len(names)), values, color='steelblue')

    # 在柱子上显示数值
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def run_full_analysis():
    """运行完整分析"""
    import torch

    print("=" * 60)
    print("Text Classification - Result Analysis")
    print("=" * 60)

    # 定义要分析的数据文件和模型
    data_files = ['bow_data.pkl', 'bigram_data.pkl', 'trigram_data.pkl']
    model_types = ['softmax', 'perceptron']
    learning_rates = [0.001, 0.01, 0.1]

    # 存储结果
    results = defaultdict(dict)
    all_histories = {}
    all_predictions = {}
    all_true_labels = {}

    # 加载所有数据和模型
    for data_file in data_files:
        data_name = data_file.replace('.pkl', '')
        X_test, y_test, vocab_size = load_data(data_file)
        all_true_labels[data_name] = y_test.numpy()

        print(f"\n{'='*60}")
        print(f"Analyzing {data_name} (vocab size: {vocab_size})")
        print(f"{'='*60}")

        for model_type in model_types:
            for lr in learning_rates:
                # 模型文件名
                model_file = f"{model_type}_{data_name}_lr{lr}.pt"
                history_file = f"{model_type}_{data_name}_lr{lr}_history.pkl"

                model_path = os.path.join(MODELS_DIR, model_file)
                history_path = os.path.join(MODELS_DIR, history_file)

                if not os.path.exists(model_path):
                    print(f"  {model_type} lr={lr}: Model not found, skipping...")
                    continue

                # 加载模型
                if model_type == 'softmax':
                    model = SoftmaxClassifier.load(model_path, lr)
                else:
                    model = PerceptronClassifier.load(model_path, lr)

                # 计算测试准确率
                test_acc = model.accuracy(X_test, y_test)

                # 获取预测结果
                predictions = model.predict(X_test).numpy()
                key = f"{model_type}_{data_name}_lr{lr}"
                all_predictions[key] = predictions

                results[data_name][key] = test_acc
                print(f"  {model_type} lr={lr}: Test Acc = {test_acc:.4f}")

                # 加载训练历史
                if os.path.exists(history_path):
                    history = load_history(history_file)
                    all_histories[key] = history

    # 创建图表输出目录
    os.makedirs(os.path.join(DOCS_DIR, 'figures'), exist_ok=True)

    # 1. 绘制 Bag of Words vs N-gram 对比图
    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    print("=" * 60)

    # 提取 softmax + lr=0.01 的结果进行对比
    comparison_results = {}
    for data_name in results:
        key = f"softmax_{data_name}_lr0.01"
        if key in results.get(data_name, {}):
            comparison_results[data_name] = results[data_name][key]

    if comparison_results:
        fig = plot_comparison_bar(comparison_results, "Bag of Words vs N-gram Comparison (Softmax, lr=0.01)")
        fig.savefig(os.path.join(DOCS_DIR, 'figures', 'bow_vs_ngram.png'), dpi=150)
        print("Saved: bow_vs_ngram.png")
        plt.close(fig)

    # 2. 绘制不同学习率对比图
    for data_name in results:
        lr_results = {}
        for key, acc in results[data_name].items():
            if 'softmax' in key:
                lr = key.split('_lr')[-1]
                lr_results[lr] = acc

        if lr_results:
            fig = plot_comparison_bar(lr_results, f"Learning Rate Comparison - {data_name} (Softmax)")
            fig.savefig(os.path.join(DOCS_DIR, 'figures', f'lr_comparison_{data_name}.png'), dpi=150)
            print(f"Saved: lr_comparison_{data_name}.png")
            plt.close(fig)

    # 3. 绘制 Softmax vs Perceptron 对比
    model_comparison = {}
    for data_name in results:
        for key, acc in results[data_name].items():
            model_name = key.split('_')[0]
            if model_name not in model_comparison:
                model_comparison[model_name] = []
            model_comparison[model_name].append(acc)

    if model_comparison:
        # 计算平均准确率
        avg_results = {k: np.mean(v) for k, v in model_comparison.items()}
        fig = plot_comparison_bar(avg_results, "Softmax vs Perceptron (Average Accuracy)")
        fig.savefig(os.path.join(DOCS_DIR, 'figures', 'softmax_vs_perceptron.png'), dpi=150)
        print("Saved: softmax_vs_perceptron.png")
        plt.close(fig)

    # 4. 绘制训练曲线
    for data_name in results:
        # 只绘制 softmax lr=0.01 的训练曲线
        key = f"softmax_{data_name}_lr0.01"
        if key in all_histories:
            fig = plot_training_curves({key: all_histories[key]}, 'val_acc',
                                       f"Training Curve - {data_name}")
            fig.savefig(os.path.join(DOCS_DIR, 'figures', f'training_curve_{data_name}.png'), dpi=150)
            print(f"Saved: training_curve_{data_name}.png")
            plt.close(fig)

    # 5. 绘制混淆矩阵（使用最佳模型）
    best_model_key = max(all_predictions.keys(),
                        key=lambda k: results[k.split('_')[1]].get(k, 0))
    if best_model_key in all_predictions:
        cm = compute_confusion_matrix(all_true_labels['bow_data'],
                                     all_predictions[best_model_key])
        class_names = ['0', '1', '2', '3', '4']
        fig = plot_confusion_matrix(cm, class_names,
                                    f"Confusion Matrix - {best_model_key}")
        fig.savefig(os.path.join(DOCS_DIR, 'figures', 'confusion_matrix.png'), dpi=150)
        print("Saved: confusion_matrix.png")
        plt.close(fig)

    # 6. 生成汇总报告
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    print("\n1. Feature Extraction Comparison (Softmax, lr=0.01):")
    print("-" * 60)
    for data_name in ['bow_data', 'bigram_data', 'trigram_data']:
        key = f"softmax_{data_name}_lr0.01"
        if key in results.get(data_name, {}):
            print(f"  {data_name:12s}: {results[data_name][key]:.4f}")

    print("\n2. Model Type Comparison (Average):")
    print("-" * 60)
    for model_type in ['softmax', 'perceptron']:
        accs = []
        for data_name in results:
            for key, acc in results[data_name].items():
                if model_type in key:
                    accs.append(acc)
        if accs:
            print(f"  {model_type:12s}: {np.mean(accs):.4f}")

    print("\n3. Learning Rate Impact:")
    print("-" * 60)
    for lr in ['0.001', '0.01', '0.1']:
        accs = []
        for data_name in results:
            for key, acc in results[data_name].items():
                if f'_lr{lr}' in key:
                    accs.append(acc)
        if accs:
            print(f"  lr={lr:6s}: {np.mean(accs):.4f}")

    print("\n4. Best Model:")
    print("-" * 60)
    best_acc = 0
    best_key = ""
    for data_name in results:
        for key, acc in results[data_name].items():
            if acc > best_acc:
                best_acc = acc
                best_key = key
    print(f"  {best_key}: {best_acc:.4f}")

    print("\n" + "=" * 60)
    print("Analysis complete! Figures saved to docs/figures/")
    print("=" * 60)

    # 保存结果到文件
    with open(os.path.join(DOCS_DIR, 'results_summary.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TEXT CLASSIFICATION - RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. Feature Extraction Comparison (Softmax, lr=0.01):\n")
        f.write("-" * 60 + "\n")
        for data_name in ['bow_data', 'bigram_data', 'trigram_data']:
            key = f"softmax_{data_name}_lr0.01"
            if key in results.get(data_name, {}):
                f.write(f"  {data_name:12s}: {results[data_name][key]:.4f}\n")

        f.write("\n2. Model Type Comparison (Average):\n")
        f.write("-" * 60 + "\n")
        for model_type in ['softmax', 'perceptron']:
            accs = []
            for data_name in results:
                for key, acc in results[data_name].items():
                    if model_type in key:
                        accs.append(acc)
            if accs:
                f.write(f"  {model_type:12s}: {np.mean(accs):.4f}\n")

        f.write("\n3. Learning Rate Impact:\n")
        f.write("-" * 60 + "\n")
        for lr in ['0.001', '0.01', '0.1']:
            accs = []
            for data_name in results:
                for key, acc in results[data_name].items():
                    if f'_lr{lr}' in key:
                        accs.append(acc)
            if accs:
                f.write(f"  lr={lr:6s}: {np.mean(accs):.4f}\n")

        f.write("\n4. Best Model:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  {best_key}: {best_acc:.4f}\n")

    print("Results summary saved to docs/results_summary.txt")


if __name__ == "__main__":
    run_full_analysis()
