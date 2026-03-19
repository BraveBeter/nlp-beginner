"""
训练器模块 - 提供通用的训练和评估功能

支持:
- Transformer (Encoder-Decoder) 训练
- GPT (Decoder-only) 训练
- 自动验证和早停
- 模型检查点保存和加载
- 训练日志记录
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable, Any
import os
import json
import time
from datetime import datetime
from pathlib import Path

from .metrics import calculate_accuracy, calculate_perplexity


class Trainer:
    """
    通用训练器类 - 支持各种序列到序列模型

    支持两种训练模式:
    1. Encoder-Decoder 模式 (Transformer)
    2. Decoder-only 模式 (GPT)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        model_type: str = "transformer",  # "transformer" or "gpt"
        checkpoint_dir: str = "outputs/models",
        log_dir: str = "outputs/logs",
        bos_idx: int = 0,
        eos_idx: int = 0,
        pad_idx: int = 0,
        max_grad_norm: float = 1.0,
    ):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 训练设备
            model_type: 模型类型 ("transformer" or "gpt")
            checkpoint_dir: 模型检查点保存目录
            log_dir: 日志保存目录
            bos_idx: 句子开始标记索引
            eos_idx: 句子结束标记索引
            pad_idx: 填充标记索引
            max_grad_norm: 梯度裁剪的最大范数
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.model_type = model_type
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.max_grad_norm = max_grad_norm

        # 创建目录
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        # 训练历史
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'epoch_times': [],
            'learning_rates': [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个 epoch

        Returns:
            平均损失, 平均准确率
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.train_loader)

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # 根据模型类型处理不同的 batch 格式
            if self.model_type == "transformer":
                # Encoder-Decoder: (src, src_key_padding_mask, tgt, tgt_key_padding_mask)
                src, src_key_padding_mask, tgt, tgt_key_padding_mask = batch
                src, src_key_padding_mask, tgt, tgt_key_padding_mask = (
                    src.to(self.device),
                    src_key_padding_mask.to(self.device),
                    tgt.to(self.device),
                    tgt_key_padding_mask.to(self.device),
                )

                # 生成因果掩码
                tgt_seq_len = tgt.size(1)
                tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=self.device), diagonal=1).bool()

                # 前向传播
                self.optimizer.zero_grad()
                output = self.model(
                    src, tgt,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )

                # 计算损失 (忽略 padding)
                # output: [batch_size, tgt_len, vocab_size]
                # tgt: [batch_size, tgt_len]
                # 我们需要预测位置 t+1 的 token，所以去掉最后一个位置的输出和第一个位置的目标
                output = output[:, :-1, :].contiguous()  # [batch_size, tgt_len-1, vocab_size]
                output = output.reshape(-1, output.shape[-1])  # [batch_size * (tgt_len-1), vocab_size]
                tgt_flat = tgt[:, 1:].reshape(-1)  # [batch_size * (tgt_len-1),] 去掉 <BOS>
                loss = self.criterion(output, tgt_flat)

            elif self.model_type == "gpt":
                # Decoder-only: (x, key_padding_mask)
                x, key_padding_mask = batch
                x, key_padding_mask = x.to(self.device), key_padding_mask.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                logits = self.model(x, key_padding_mask=key_padding_mask)

                # 计算损失 (预测下一个 token)
                logits = logits[:, :-1].contiguous()  # 去掉最后一个预测
                logits = logits.view(-1, logits.shape[-1])
                targets = x[:, 1:].contiguous()  # 去掉第一个 token
                targets = targets.view(-1)

                # 计算损失
                loss = self.criterion(logits, targets)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            # 更新参数
            self.optimizer.step()

            # 计算准确率
            if self.model_type == "transformer":
                predictions = output.argmax(dim=-1)
                accuracy = calculate_accuracy(predictions, tgt_flat, ignore_idx=self.pad_idx)
            else:
                predictions = logits.argmax(dim=-1)
                accuracy = calculate_accuracy(predictions, targets, ignore_idx=self.pad_idx)

            total_loss += loss.item()
            total_accuracy += accuracy

            self.global_step += 1

            # 打印进度
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_acc = total_accuracy / (batch_idx + 1)
                print(
                    f"Batch [{batch_idx + 1}/{num_batches}], "
                    f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}"
                )

        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        return avg_loss, avg_accuracy, epoch_time

    def validate(self) -> Tuple[float, float]:
        """
        在验证集上评估模型

        Returns:
            平均损失, 平均准确率
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                if self.model_type == "transformer":
                    src, src_key_padding_mask, tgt, tgt_key_padding_mask = batch
                    src, src_key_padding_mask, tgt, tgt_key_padding_mask = (
                        src.to(self.device),
                        src_key_padding_mask.to(self.device),
                        tgt.to(self.device),
                        tgt_key_padding_mask.to(self.device),
                    )

                    # 生成因果掩码
                    tgt_seq_len = tgt.size(1)
                    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=self.device), diagonal=1).bool()

                    output = self.model(
                        src, tgt,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=src_key_padding_mask
                    )
                    output = output[:, :-1, :].contiguous()
                    output = output.reshape(-1, output.shape[-1])
                    tgt_flat = tgt[:, 1:].reshape(-1)
                    loss = self.criterion(output, tgt_flat)
                    predictions = output.argmax(dim=-1)
                    accuracy = calculate_accuracy(predictions, tgt_flat, ignore_idx=self.pad_idx)

                elif self.model_type == "gpt":
                    x, key_padding_mask = batch
                    x, key_padding_mask = x.to(self.device), key_padding_mask.to(self.device)

                    logits = self.model(x, key_padding_mask=key_padding_mask)
                    logits = logits[:, :-1].contiguous()
                    logits = logits.view(-1, logits.shape[-1])
                    targets = x[:, 1:].contiguous()
                    targets = targets.view(-1)

                    mask_flat = key_padding_mask[:, 1:].contiguous().view(-1)
                    loss = self.criterion(logits, targets)
                    loss = loss * mask_flat
                    loss = loss.sum() / mask_flat.sum()
                    predictions = logits.argmax(dim=-1)
                    accuracy = calculate_accuracy(predictions, targets, ignore_idx=self.pad_idx)

                total_loss += loss.item()
                total_accuracy += accuracy

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        return avg_loss, avg_accuracy

    def train(
        self,
        num_epochs: int,
        patience: int = 5,
        min_delta: float = 1e-4,
        save_best_only: bool = True,
    ) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            num_epochs: 训练轮数
            patience: 早停的耐心值
            min_delta: 最小改善阈值
            save_best_only: 是否只保存最佳模型

        Returns:
            训练历史字典
        """
        print(f"开始训练 {self.model_type.upper()} 模型...")
        print(f"设备: {self.device}")
        print(f"训练批次数: {len(self.train_loader)}")
        print(f"验证批次数: {len(self.val_loader)}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print("-" * 60)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            # 训练
            train_loss, train_acc, epoch_time = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate()

            # 记录历史
            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val_loss)
            self.history['val_accuracies'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # 打印结果
            print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"时间: {epoch_time:.2f}s")

            # 保存检查点
            is_best = val_loss < best_val_loss - min_delta
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"✓ 验证损失改善: {val_loss:.4f}")

                if save_best_only:
                    self.save_checkpoint(
                        os.path.join(
                            self.checkpoint_dir,
                            f"{self.model_type}_best.pt"
                        ),
                        is_best=True
                    )
            else:
                patience_counter += 1
                print(f"验证损失未改善 (计数: {patience_counter}/{patience})")

            # 定期保存
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    os.path.join(
                        self.checkpoint_dir,
                        f"{self.model_type}_epoch_{epoch + 1}.pt"
                    )
                )

            # 早停检查
            if patience_counter >= patience:
                print(f"\n早停触发！验证损失在 {patience} 轮内未改善。")
                break

            # 保存训练历史
            self.save_history()

        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"训练轮数: {self.current_epoch}")
        print("=" * 60)

        return self.history

    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """
        保存模型检查点

        Args:
            filepath: 保存路径
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'model_type': self.model_type,
        }

        torch.save(checkpoint, filepath)
        if is_best:
            print(f"✓ 最佳模型已保存至: {filepath}")

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        加载模型检查点

        Args:
            filepath: 检查点文件路径

        Returns:
            检查点字典
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        print(f"✓ 检查点已加载: {filepath}")
        print(f"  - Epoch: {self.current_epoch}")
        print(f"  - Best Val Loss: {self.best_val_loss:.4f}")

        return checkpoint

    def save_history(self):
        """保存训练历史到 JSON 文件"""
        history_path = self.log_dir / f"{self.model_type}_history.json"

        # 转换 numpy 类型为 Python 原生类型
        history_serializable = {}
        for key, values in self.history.items():
            history_serializable[key] = [
                float(v) if v is not None else None for v in values
            ]

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_serializable, f, indent=2, ensure_ascii=False)

        print(f"✓ 训练历史已保存至: {history_path}")


class Evaluator:
    """
    评估器类 - 用于模型评估和对比
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        model_type: str = "transformer",
        bos_idx: int = 0,
        eos_idx: int = 0,
        pad_idx: int = 0,
    ):
        """
        初始化评估器

        Args:
            model: 要评估的模型
            device: 计算设备
            model_type: 模型类型
            bos_idx: 句子开始标记索引
            eos_idx: 句子结束标记索引
            pad_idx: 填充标记索引
        """
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    def evaluate(
        self,
        test_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        在测试集上评估模型

        Args:
            test_loader: 测试数据加载器
            criterion: 可选的损失函数

        Returns:
            评估指标字典
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_seq_accuracy = 0.0
        num_batches = len(test_loader)

        with torch.no_grad():
            for batch in test_loader:
                if self.model_type == "transformer":
                    src, src_key_padding_mask, tgt, tgt_key_padding_mask = batch
                    src, src_key_padding_mask, tgt, tgt_key_padding_mask = (
                        src.to(self.device),
                        src_key_padding_mask.to(self.device),
                        tgt.to(self.device),
                        tgt_key_padding_mask.to(self.device),
                    )

                    # 生成因果掩码
                    tgt_seq_len = tgt.size(1)
                    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=self.device), diagonal=1).bool()

                    output = self.model(
                        src, tgt,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=src_key_padding_mask
                    )
                    output = output[:, :-1, :].contiguous()
                    output = output.reshape(-1, output.shape[-1])
                    tgt_flat = tgt[:, 1:].reshape(-1)

                    if criterion is not None:
                        loss = criterion(output, tgt_flat)
                        total_loss += loss.item()

                    predictions = output.argmax(dim=-1)
                    accuracy = calculate_accuracy(predictions, tgt_flat, ignore_idx=self.pad_idx)
                    total_accuracy += accuracy

                    # 序列准确率 (完全匹配)
                    predictions = output.argmax(dim=-1).view(
                        tgt.size(0), -1
                    )
                    targets = tgt[:, 1:]
                    seq_acc = self._sequence_accuracy(
                        predictions, targets, self.eos_idx
                    )
                    total_seq_accuracy += seq_acc

                elif self.model_type == "gpt":
                    x, key_padding_mask = batch
                    x, key_padding_mask = x.to(self.device), key_padding_mask.to(self.device)

                    logits = self.model(x, key_padding_mask=key_padding_mask)
                    logits = logits[:, :-1].contiguous()
                    logits = logits.view(-1, logits.shape[-1])
                    targets = x[:, 1:].contiguous()
                    targets = targets.view(-1)

                    if criterion is not None:
                        loss = criterion(logits, targets)
                        total_loss += loss.item()

                    predictions = logits.argmax(dim=-1)
                    accuracy = calculate_accuracy(predictions, targets, ignore_idx=self.pad_idx)
                    total_accuracy += accuracy

                    # 序列准确率
                    predictions = logits.argmax(dim=-1).view(x.size(0), -1)
                    targets = x[:, 1:]
                    seq_acc = self._sequence_accuracy(
                        predictions, targets, self.eos_idx
                    )
                    total_seq_accuracy += seq_acc

        results = {
            'test_loss': total_loss / num_batches if criterion is not None else None,
            'token_accuracy': total_accuracy / num_batches,
            'sequence_accuracy': total_seq_accuracy / num_batches,
        }

        return results

    def _sequence_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        eos_idx: int,
    ) -> float:
        """
        计算序列准确率（完全匹配的比例）

        Args:
            predictions: 预测序列 [batch_size, seq_len]
            targets: 目标序列 [batch_size, seq_len]
            eos_idx: 结束标记索引

        Returns:
            序列准确率
        """
        batch_size = predictions.size(0)
        correct = 0

        for i in range(batch_size):
            # 找到 EOS 位置
            pred_seq = predictions[i]
            target_seq = targets[i]

            # 比较（到 EOS 为止）
            pred_eos_pos = (pred_seq == eos_idx).nonzero(as_tuple=True)
            target_eos_pos = (target_seq == eos_idx).nonzero(as_tuple=True)

            if len(pred_eos_pos[0]) > 0:
                pred_end = pred_eos_pos[0][0].item()
            else:
                pred_end = len(pred_seq)

            if len(target_eos_pos[0]) > 0:
                target_end = target_eos_pos[0][0].item()
            else:
                target_end = len(target_seq)

            # 比较
            min_len = min(pred_end, target_end)
            if torch.equal(pred_seq[:min_len], target_seq[:min_len]):
                correct += 1

        return correct / batch_size

    def generate_samples(
        self,
        test_loader: DataLoader,
        num_samples: int = 5,
        max_length: int = 50,
    ) -> List[Tuple[str, str, str]]:
        """
        生成样本进行可视化

        Args:
            test_loader: 测试数据加载器
            num_samples: 生成样本数量
            max_length: 最大生成长度

        Returns:
            (输入, 目标, 预测) 元组列表
        """
        self.model.eval()
        samples = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_samples:
                    break

                if self.model_type == "transformer":
                    src, src_key_padding_mask, tgt, _ = batch
                    src, src_key_padding_mask = (
                        src.to(self.device),
                        src_key_padding_mask.to(self.device),
                    )

                    # 生成
                    output = self.model.generate(
                        src,
                        max_len=max_length,
                        bos_idx=self.bos_idx,
                        eos_idx=self.eos_idx,
                        src_key_padding_mask=src_key_padding_mask,
                    )

                    # 转换为文本 (这里需要 tokenizer，暂时返回索引)
                    samples.append((
                        src[0].cpu().tolist(),
                        tgt[0].cpu().tolist(),
                        output[0].cpu().tolist(),
                    ))

                elif self.model_type == "gpt":
                    x, mask = batch
                    x, mask = x.to(self.device), mask.to(self.device)

                    # 生成
                    output = self.model.generate(
                        x[:, :1],  # 只用第一个 token 作为提示
                        max_len=max_length,
                        eos_idx=self.eos_idx,
                    )

                    samples.append((
                        x[0].cpu().tolist(),
                        x[0].cpu().tolist(),  # 对于 GPT，输入和目标相同
                        output[0].cpu().tolist(),
                    ))

        return samples


def compare_models(
    model1: nn.Module,
    model2: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    model_type: str = "transformer",
) -> Dict[str, Dict[str, float]]:
    """
    比较两个模型的性能

    Args:
        model1: 第一个模型
        model2: 第二个模型
        test_loader: 测试数据加载器
        device: 计算设备
        model1_name: 第一个模型名称
        model2_name: 第二个模型名称
        model_type: 模型类型

    Returns:
        比较结果字典
    """
    print(f"\n比较 {model1_name} vs {model2_name}")
    print("=" * 60)

    # 评估第一个模型
    evaluator1 = Evaluator(model1, device, model_type)
    results1 = evaluator1.evaluate(test_loader)

    # 评估第二个模型
    evaluator2 = Evaluator(model2, device, model_type)
    results2 = evaluator2.evaluate(test_loader)

    # 打印结果
    print(f"\n{model1_name}:")
    print(f"  - Token Accuracy: {results1['token_accuracy']:.4f}")
    print(f"  - Sequence Accuracy: {results1['sequence_accuracy']:.4f}")

    print(f"\n{model2_name}:")
    print(f"  - Token Accuracy: {results2['token_accuracy']:.4f}")
    print(f"  - Sequence Accuracy: {results2['sequence_accuracy']:.4f}")

    # 计算差异
    token_diff = results2['token_accuracy'] - results1['token_accuracy']
    seq_diff = results2['sequence_accuracy'] - results1['sequence_accuracy']

    print(f"\n差异 ({model2_name} - {model1_name}):")
    print(f"  - Token Accuracy: {token_diff:+.4f}")
    print(f"  - Sequence Accuracy: {seq_diff:+.4f}")

    return {
        model1_name: results1,
        model2_name: results2,
    }