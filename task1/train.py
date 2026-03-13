"""
模型训练脚本
实现 Softmax 多分类和感知机多分类
"""

import os
import pickle
import argparse
import numpy as np
import torch
from typing import Tuple, Dict

# 设置路径
TEMP_DATA_DIR = "temp_data"
MODELS_DIR = "models"

# 设置随机种子以保证可复现性
torch.manual_seed(42)
np.random.seed(42)


class SoftmaxClassifier:
    """
    Softmax 多分类器
    使用交叉熵损失
    """

    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        self.W = torch.randn(input_dim, num_classes) * 0.01
        self.b = torch.zeros(num_classes)

        # 需要梯度的参数
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算 logits
        X: (batch_size, input_dim)
        """
        return X @ self.W + self.b

    def softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算 softmax 概率
        logits: (batch_size, num_classes)
        """
        # 数值稳定性：减去最大值
        max_vals = logits.max(dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits - max_vals)
        sum_exp = exp_logits.sum(dim=1, keepdim=True)
        return exp_logits / sum_exp

    def cross_entropy_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        交叉熵损失
        logits: (batch_size, num_classes)
        y: (batch_size,)
        """
        # 数值稳定的交叉熵
        max_vals = logits.max(dim=1, keepdim=True)[0]
        log_sum_exp = torch.log(torch.exp(logits - max_vals).sum(dim=1, keepdim=True)) + max_vals

        # 选择正确类别的 logit
        correct_logits = logits[range(len(y)), y]

        loss = (log_sum_exp.squeeze() - correct_logits).mean()
        return loss

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        训练一步
        """
        # 前向传播
        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)

        # 反向传播
        loss.backward()

        # 更新参数
        with torch.no_grad():
            self.W -= self.learning_rate * self.W.grad
            self.b -= self.learning_rate * self.b.grad

        # 清零梯度
        self.W.grad.zero_()
        self.b.grad.zero_()

        return loss.item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        预测
        """
        logits = self.forward(X)
        return logits.argmax(dim=1)

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        计算准确率
        """
        preds = self.predict(X)
        return (preds == y).float().mean().item()

    def save(self, filepath: str):
        """
        保存模型
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save({
            'W': self.W.detach().cpu(),
            'b': self.b.detach().cpu(),
            'input_dim': self.input_dim,
            'num_classes': self.num_classes
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, learning_rate: float = 0.01):
        """
        加载模型
        """
        checkpoint = torch.load(filepath)
        model = cls(checkpoint['input_dim'], checkpoint['num_classes'], learning_rate)
        model.W.data = checkpoint['W']
        model.b.data = checkpoint['b']
        return model


class PerceptronClassifier:
    """
    感知机多分类器
    使用感知机损失
    """

    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        self.W = torch.randn(input_dim, num_classes) * 0.01
        self.b = torch.zeros(num_classes)

        # 需要梯度的参数
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算 scores
        """
        return X @ self.W + self.b

    def perceptron_loss(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        感知机损失：hinge loss 的变体
        对于正确类别的分数，希望它比其他类别高
        scores: (batch_size, num_classes)
        y: (batch_size,)
        """
        batch_size = scores.shape[0]

        # 获取正确类别的分数
        correct_scores = scores[range(batch_size), y].unsqueeze(1)  # (batch_size, 1)

        # 感知机损失：max(0, s_j - s_y + margin)
        # margin 设为 1
        margin = 1.0
        losses = torch.clamp(scores - correct_scores + margin, min=0)

        # 不计算正确类别的损失（它总是 margin）
        losses[range(batch_size), y] = 0

        return losses.sum() / batch_size

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        训练一步
        """
        # 前向传播
        scores = self.forward(X)
        loss = self.perceptron_loss(scores, y)

        # 反向传播
        loss.backward()

        # 更新参数
        with torch.no_grad():
            self.W -= self.learning_rate * self.W.grad
            self.b -= self.learning_rate * self.b.grad

        # 清零梯度
        self.W.grad.zero_()
        self.b.grad.zero_()

        return loss.item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        预测
        """
        scores = self.forward(X)
        return scores.argmax(dim=1)

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        计算准确率
        """
        preds = self.predict(X)
        return (preds == y).float().mean().item()

    def save(self, filepath: str):
        """
        保存模型
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save({
            'W': self.W.detach().cpu(),
            'b': self.b.detach().cpu(),
            'input_dim': self.input_dim,
            'num_classes': self.num_classes
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, learning_rate: float = 0.01):
        """
        加载模型
        """
        checkpoint = torch.load(filepath)
        model = cls(checkpoint['input_dim'], checkpoint['num_classes'], learning_rate)
        model.W.data = checkpoint['W']
        model.b.data = checkpoint['b']
        return model


def load_processed_data(filename: str) -> Dict:
    """
    加载处理后的数据
    """
    filepath = os.path.join(TEMP_DATA_DIR, filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def train_model(model_type: str, data_file: str, learning_rate: float,
                epochs: int, batch_size: int = 32, device: str = 'cpu') -> Dict:
    """
    训练模型
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type} with {data_file}")
    print(f"Learning rate: {learning_rate}, Epochs: {epochs}, Batch size: {batch_size}")
    print(f"{'='*60}")

    # 加载数据
    data = load_processed_data(data_file)
    X_train_np, y_train_np = data['X_train'], data['y_train']
    X_test_np, y_test_np = data['X_test'], data['y_test']

    # 划分训练集和验证集（80%-20%）
    num_train = len(X_train_np)
    num_val = num_train // 5
    num_train_actual = num_train - num_val

    indices = np.random.permutation(num_train)
    train_indices = indices[:num_train_actual]
    val_indices = indices[num_train_actual:]

    X_val_np, y_val_np = X_train_np[val_indices], y_train_np[val_indices]
    X_train_np, y_train_np = X_train_np[train_indices], y_train_np[train_indices]

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val_np, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val_np, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np, dtype=torch.long).to(device)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train_np))

    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 初始化模型
    if model_type == 'softmax':
        model = SoftmaxClassifier(input_dim, num_classes, learning_rate)
    else:
        model = PerceptronClassifier(input_dim, num_classes, learning_rate)

    # 训练历史
    history = {
        'train_loss': [],
        'val_acc': [],
        'test_acc': []
    }

    # 训练循环
    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        epoch_loss = 0.0

        # 每个 epoch 内打乱数据
        perm = torch.randperm(len(X_train))

        for i in range(num_batches):
            # 获取 batch
            indices = perm[i * batch_size:(i + 1) * batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            # 训练一步
            loss = model.train_step(X_batch, y_batch)
            epoch_loss += loss

        # 处理剩余数据
        if num_batches * batch_size < len(X_train):
            indices = perm[num_batches * batch_size:]
            X_batch = X_train[indices]
            y_batch = y_train[indices]
            loss = model.train_step(X_batch, y_batch)
            epoch_loss += loss

        avg_loss = epoch_loss / (num_batches + (1 if num_batches * batch_size < len(X_train) else 0))

        # 验证
        val_acc = model.accuracy(X_val, y_val)
        test_acc = model.accuracy(X_test, y_test)

        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f} - Test Acc: {test_acc:.4f}")

    # 保存模型
    model_name = f"{model_type}_{data_file.replace('.pkl', '')}_lr{learning_rate}.pt"
    model_path = os.path.join(MODELS_DIR, model_name)
    model.save(model_path)

    return history


def main():
    parser = argparse.ArgumentParser(description='Train text classification models')
    parser.add_argument('--model', type=str, default='softmax',
                        choices=['softmax', 'perceptron'],
                        help='Model type')
    parser.add_argument('--data', type=str, default='bow_data.pkl',
                        choices=['bow_data.pkl', 'unigram_data.pkl', 'bigram_data.pkl', 'trigram_data.pkl'],
                        help='Data file')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')

    args = parser.parse_args()

    # 训练模型
    history = train_model(
        model_type=args.model,
        data_file=args.data,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    # 保存训练历史
    history_name = f"{args.model}_{args.data.replace('.pkl', '')}_lr{args.lr}_history.pkl"
    history_path = os.path.join(MODELS_DIR, history_name)
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
