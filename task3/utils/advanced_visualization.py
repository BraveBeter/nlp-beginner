"""
高级可视化工具 - Phase 4 实验可视化

提供：
1. 训练曲线绘制（损失、准确率）
2. 多模型对比
3. 注意力热力图
4. 消融实验对比
5. 综合分析报告生成
================================================================================
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch
import torch.nn as nn
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
rcParams['axes.unicode_minus'] = False

# 设置图表风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


class TrainingVisualizer:
    """
    训练可视化器

    功能：
    1. 绘制训练/验证损失曲线
    2. 绘制准确率曲线
    3. 对比多个模型的训练过程
    4. 生成训练报告
    """

    def __init__(self, output_dir: str = "outputs/figures"):
        """
        初始化可视化器

        Args:
            output_dir: 图表保存目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 颜色方案
        self.colors = {
            'transformer': '#1f77b4',  # 蓝色
            'gpt': '#ff7f0e',          # 橙色
            'train': '#2ca02c',        # 绿色
            'val': '#d62728',          # 红色
            'test': '#9467bd',         # 紫色
        }

        self.markers = {
            'transformer': 'o',
            'gpt': 's',
        }

    def load_training_history(self, history_path: str) -> Dict[str, List[float]]:
        """
        加载训练历史

        Args:
            history_path: 历史文件路径

        Returns:
            训练历史字典
        """
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        return history

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        model_name: str,
        save: bool = True
    ) -> plt.Figure:
        """
        绘制单个模型的训练曲线

        Args:
            history: 训练历史
            model_name: 模型名称
            save: 是否保存图表

        Returns:
            matplotlib Figure 对象
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = range(1, len(history['train_losses']) + 1)

        # 1. 损失曲线
        ax1 = axes[0]
        ax1.plot(epochs, history['train_losses'],
                label='Train Loss', color=self.colors['train'], marker='o', markersize=4)
        ax1.plot(epochs, history['val_losses'],
                label='Val Loss', color=self.colors['val'], marker='s', markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_name}: Training & Validation Loss',
                    fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 标注最佳点
        best_epoch = np.argmin(history['val_losses']) + 1
        best_loss = np.min(history['val_losses'])
        ax1.scatter([best_epoch], [best_loss], color='red', s=100, zorder=5)
        ax1.annotate(f'Best: {best_loss:.4f}',
                    xy=(best_epoch, best_loss),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

        # 2. 准确率曲线
        ax2 = axes[1]
        ax2.plot(epochs, history['val_accuracies'],
                label='Val Accuracy', color=self.colors['test'], marker='^', markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title(f'{model_name}: Validation Accuracy',
                    fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.0)

        # 标注最佳点
        best_acc_epoch = np.argmax(history['val_accuracies']) + 1
        best_acc = np.max(history['val_accuracies'])
        ax2.scatter([best_acc_epoch], [best_acc], color='red', s=100, zorder=5)
        ax2.annotate(f'Best: {best_acc:.4f}',
                    xy=(best_acc_epoch, best_acc),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

        # 3. 训练时间
        ax3 = axes[2]
        if 'epoch_times' in history:
            ax3.bar(epochs, history['epoch_times'], color='steelblue', alpha=0.7)
            ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
            ax3.set_title(f'{model_name}: Training Time per Epoch',
                        fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')

            # 添加平均时间线
            avg_time = np.mean(history['epoch_times'])
            ax3.axhline(y=avg_time, color='red', linestyle='--', linewidth=2)
            ax3.text(0.5, avg_time * 1.1, f'Avg: {avg_time:.2f}s',
                    fontsize=10, color='red', fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"{model_name}_training_curves.png"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"✓ 训练曲线已保存: {save_path}")

        return fig

    def compare_models(
        self,
        histories: Dict[str, Dict[str, List[float]]],
        save: bool = True
    ) -> plt.Figure:
        """
        对比多个模型的训练过程

        Args:
            histories: 模型名称到训练历史的映射
            save: 是否保存图表

        Returns:
            matplotlib Figure 对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 训练损失对比
        ax1 = axes[0, 0]
        for model_name, history in histories.items():
            epochs = range(1, len(history['train_losses']) + 1)
            color = self.colors.get(model_name.split('_')[0], '#333333')
            marker = self.markers.get(model_name.split('_')[0], 'o')
            ax1.plot(epochs, history['train_losses'],
                    label=model_name, color=color, marker=marker, markersize=3)

        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. 验证损失对比
        ax2 = axes[0, 1]
        for model_name, history in histories.items():
            epochs = range(1, len(history['val_losses']) + 1)
            color = self.colors.get(model_name.split('_')[0], '#333333')
            marker = self.markers.get(model_name.split('_')[0], 'o')
            ax2.plot(epochs, history['val_losses'],
                    label=model_name, color=color, marker=marker, markersize=3)

        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 3. 验证准确率对比
        ax3 = axes[1, 0]
        for model_name, history in histories.items():
            epochs = range(1, len(history['val_accuracies']) + 1)
            color = self.colors.get(model_name.split('_')[0], '#333333')
            marker = self.markers.get(model_name.split('_')[0], 'o')
            ax3.plot(epochs, history['val_accuracies'],
                    label=model_name, color=color, marker=marker, markersize=3)

        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.0)

        # 4. 最终性能对比
        ax4 = axes[1, 1]

        models = list(histories.keys())
        final_losses = [histories[m]['val_losses'][-1] for m in models]
        final_accs = [histories[m]['val_accuracies'][-1] for m in models]

        x = np.arange(len(models))
        width = 0.35

        ax4_twin = ax4.twinx()

        bars1 = ax4.bar(x - width/2, final_losses, width,
                       label='Final Loss', color='#d62728', alpha=0.7)
        bars2 = ax4_twin.bar(x + width/2, final_accs, width,
                            label='Final Accuracy', color='#2ca02c', alpha=0.7)

        ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Final Loss', color='#d62728', fontsize=12, fontweight='bold')
        ax4_twin.set_ylabel('Final Accuracy', color='#2ca02c', fontsize=12, fontweight='bold')
        ax4.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.legend(loc='upper left', fontsize=9)
        ax4_twin.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            save_path = self.output_dir / "model_comparison_curves.png"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"✓ 模型对比曲线已保存: {save_path}")

        return fig

    def plot_learning_curves_from_files(
        self,
        log_dir: str,
        save: bool = True
    ) -> plt.Figure:
        """
        从日志文件加载并绘制学习曲线

        Args:
            log_dir: 日志目录
            save: 是否保存图表

        Returns:
            matplotlib Figure 对象
        """
        log_dir = Path(log_dir)

        # 查找所有历史文件
        history_files = list(log_dir.glob("*_history.json"))

        if not history_files:
            raise FileNotFoundError(f"No history files found in {log_dir}")

        # 加载所有历史
        histories = {}
        for hist_file in history_files:
            model_name = hist_file.stem.replace("_history", "")
            try:
                history = self.load_training_history(str(hist_file))
                histories[model_name] = history
                print(f"✓ 加载 {model_name} 的训练历史")
            except Exception as e:
                print(f"✗ 加载 {model_name} 失败: {e}")

        if not histories:
            raise ValueError("No valid training histories found")

        # 如果只有一个模型，绘制单个模型的曲线
        if len(histories) == 1:
            model_name = list(histories.keys())[0]
            return self.plot_training_curves(histories[model_name], model_name, save)
        else:
            return self.compare_models(histories, save)

    def generate_training_report(
        self,
        log_dir: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        生成训练报告（Markdown 格式）

        Args:
            log_dir: 日志目录
            output_path: 输出文件路径

        Returns:
            报告内容
        """
        log_dir = Path(log_dir)
        history_files = list(log_dir.glob("*_history.json"))

        report_lines = [
            "# 训练报告\n",
            "## 模型训练统计\n",
            f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**日志目录**: {log_dir}\n",
            f"**模型数量**: {len(history_files)}\n",
            "\n---\n",
        ]

        # 模型对比表格
        report_lines.extend([
            "## 性能对比\n",
            "\n### 最终性能\n",
            "| 模型 | 训练轮数 | 最终训练损失 | 最终验证损失 | 最终验证准确率 |",
            "|------|----------|-------------|-------------|---------------|",
        ])

        for hist_file in history_files:
            model_name = hist_file.stem.replace("_history", "")
            try:
                history = self.load_training_history(str(hist_file))

                num_epochs = len(history['train_losses'])
                final_train_loss = history['train_losses'][-1]
                final_val_loss = history['val_losses'][-1]
                final_val_acc = history['val_accuracies'][-1]

                report_lines.append(
                    f"| {model_name} | {num_epochs} | {final_train_loss:.4f} | "
                    f"{final_val_loss:.4f} | {final_val_acc:.4f} |"
                )
            except Exception as e:
                report_lines.append(f"| {model_name} | - | - | - | - |")

        # 最佳性能
        report_lines.extend([
            "\n### 最佳性能\n",
            "| 模型 | 最佳验证损失 | 最佳验证准确率 | 出现轮次 |",
            "|------|-------------|---------------|---------|",
        ])

        for hist_file in history_files:
            model_name = hist_file.stem.replace("_history", "")
            try:
                history = self.load_training_history(str(hist_file))

                best_val_loss = min(history['val_losses'])
                best_val_loss_epoch = history['val_losses'].index(best_val_loss) + 1

                best_val_acc = max(history['val_accuracies'])
                best_val_acc_epoch = history['val_accuracies'].index(best_val_acc) + 1

                report_lines.append(
                    f"| {model_name} | {best_val_loss:.4f} (Epoch {best_val_loss_epoch}) | "
                    f"{best_val_acc:.4f} (Epoch {best_val_acc_epoch}) | - |"
                )
            except Exception as e:
                report_lines.append(f"| {model_name} | - | - | - |")

        # 训练效率
        if 'epoch_times' in history:
            report_lines.extend([
                "\n### 训练效率\n",
                "| 模型 | 总训练时间 | 平均每轮时间 |",
                "|------|-----------|-------------|",
            ])

            for hist_file in history_files:
                model_name = hist_file.stem.replace("_history", "")
                try:
                    history = self.load_training_history(str(hist_file))

                    if 'epoch_times' in history:
                        total_time = sum(history['epoch_times'])
                        avg_time = total_time / len(history['epoch_times'])

                        report_lines.append(
                            f"| {model_name} | {total_time:.2f}s | {avg_time:.2f}s |"
                        )
                except Exception as e:
                    report_lines.append(f"| {model_name} | - | - |")

        report_content = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✓ 训练报告已保存: {output_path}")

        return report_content


class AttentionVisualizer:
    """
    注意力可视化器

    功能：
    1. 提取模型的注意力权重
    2. 绘制注意力热力图
    3. 分析注意力模式
    4. 对比不同层的注意力
    """

    def __init__(self, output_dir: str = "outputs/figures"):
        """
        初始化注意力可视化器

        Args:
            output_dir: 图表保存目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_attention_weights(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        layer_idx: int = -1,
        head_idx: int = 0
    ) -> torch.Tensor:
        """
        提取模型的注意力权重

        Args:
            model: Transformer 模型
            input_ids: 输入 token IDs
            layer_idx: 层索引（-1 表示最后一层）
            head_idx: 注意力头索引

        Returns:
            注意力权重矩阵
        """
        model.eval()

        # 这是一个简化的实现
        # 实际需要在模型中注册 hook 来获取注意力权重
        # 这里提供一个框架，具体实现需要根据模型结构调整

        with torch.no_grad():
            # 前向传播并获取注意力权重
            # 注意：这需要模型支持返回注意力权重
            output = model(input_ids)

            # 这里应该返回实际的注意力权重
            # 暂时返回一个随机矩阵作为示例
            seq_len = input_ids.size(1)
            attention_weights = torch.randn(seq_len, seq_len)

        return attention_weights

    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_name: str,
        head_idx: int,
        save: bool = True
    ) -> plt.Figure:
        """
        绘制注意力热力图

        Args:
            attention_weights: 注意力权重矩阵
            tokens: token 列表
            layer_name: 层名称
            head_idx: 注意力头索引
            save: 是否保存图表

        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # 转换为 numpy 数组
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()

        # 绘制热力图
        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')

        # 设置刻度
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=12)

        ax.set_title(f'Attention Heatmap: {layer_name}, Head {head_idx}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Tokens', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"attention_{layer_name}_head{head_idx}.png"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"✓ 注意力热力图已保存: {save_path}")

        return fig

    def plot_multi_head_attention(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_name: str,
        save: bool = True
    ) -> plt.Figure:
        """
        绘制多头注意力的热力图

        Args:
            attention_weights: 多头注意力权重 [num_heads, seq_len, seq_len]
            tokens: token 列表
            layer_name: 层名称
            save: 是否保存图表

        Returns:
            matplotlib Figure 对象
        """
        num_heads = attention_weights.size(0)

        # 计算子图布局
        ncols = min(4, num_heads)
        nrows = (num_heads + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))

        if num_heads == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # 转换为 numpy 数组
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()

        for head_idx in range(num_heads):
            ax = axes[head_idx]
            attn = attention_weights[head_idx]

            im = ax.imshow(attn, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
            ax.set_title(f'Head {head_idx}', fontsize=10, fontweight='bold')

        # 隐藏多余的子图
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Multi-Head Attention: {layer_name}',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = self.output_dir / f"multihead_attention_{layer_name}.png"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"✓ 多头注意力热力图已保存: {save_path}")

        return fig


def quick_visualize_all_logs(log_dir: str = "outputs/logs"):
    """
    快速可视化所有训练日志

    Args:
        log_dir: 日志目录
    """
    print(f"\n开始可视化训练日志...")
    print("=" * 60)

    visualizer = TrainingVisualizer()

    try:
        # 绘制训练曲线
        fig = visualizer.plot_learning_curves_from_files(log_dir, save=True)

        # 生成训练报告
        report_path = "outputs/figures/training_report.md"
        report = visualizer.generate_training_report(log_dir, report_path)

        print(f"\n✓ 所有可视化完成！")
        print(f"图表保存位置: {visualizer.output_dir}")
        print(f"训练报告: {report_path}")

    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 测试可视化工具
    quick_visualize_all_logs()