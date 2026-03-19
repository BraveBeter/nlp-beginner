# 角色与目标
你现在是一位资深的 AI 算法工程师和 PyTorch 架构师。你的任务是完全从零开始（仅依赖基础的 PyTorch 张量操作和 `torch.nn` 基础组件，不允许直接调用 `nn.Transformer`）实现《Attention Is All You Need》中的标准 Transformer 架构，并基于该架构完成多位数加法和语言模型两个子任务的实验与可视化。

请确保代码具有极高的工程质量：模块化解耦、包含 Type Hint、详细的中文注释、以及优雅的变量命名。

# 第一阶段：核心架构实现 (Architecture Implementation)
请按照 PyTorch 的标准 API 逻辑构建模型，分为以下几个解耦的模块：

1. **基础组件**：
   - 实现 `MultiheadAttention`：必须手动实现 Scaled Dot-Product Attention。
   - 实现 `PositionalEncoding`：标准正弦/余弦位置编码。
   - 实现 `LayerNorm` 与残差连接（Residual Connection）。*(注：请在注释中对比 Layer Norm 与 Batch Norm 在序列任务中的适用性差异)*。
   - 实现精确的掩码机制：`padding_mask`（用于屏蔽 <PAD> token）与 `subsequent_mask`（用于 Decoder 的自回归因果掩码）。

2. **网络层堆叠**：
   - 实现 `TransformerEncoderLayer` 和 `TransformerDecoderLayer`。
   - 将 N 个 Layer 堆叠为 `TransformerEncoder` 和 `TransformerDecoder`。
   - 最终封装为一个完整的 `Transformer` 类。

3. **变体支持**：
   - 代码结构需支持轻松切换为 **Decoder-only** 架构（类似 GPT），以便后续进行对比实验。

4. **前向传播逻辑区分**：
   - **训练模式**：采用 Teacher Forcing，支持对整个 sequence 并行计算 loss。
   - **推理模式 (Inference)**：实现 `predict_next_token` 的自回归生成逻辑（支持贪心解码或简单的 Temperature Sampling）。

---

# 第二阶段：子任务实验与数据构造 (Tasks & Data)

## 子任务 1：多位数加法学习 (Multi-digit Addition)
1. **数据生成器**：
   - 编写脚本自动生成加法公式字符串（如 `"123+456=579"`）。
   - 涵盖 3+3, 3+4, 4+3, 3+5, 5+3, 4+4 的多位数加法。
2. **泛化性实验设计**：
   - **IID 划分**：随机打乱划分 Train/Test。
   - **OOD 划分（重点探究泛化性）**：例如训练集仅包含 3+3 和 4+4，测试集测试 3+4 和 4+3；或按数值大小截断划分（训练集数值较小，测试集数值较大）。
3. **模型训练**：
   - 字符级 (Character-level) Tokenization（词表包含 0-9, +, =, <PAD>, <BOS>, <EOS>）。
   - 对比标准 Encoder-Decoder 和 Decoder-only 架构在加法任务上的收敛速度和泛化准确率。

## 子任务 2：小型语言模型训练 (Language Model)
1. **语料准备**：
   - 编写脚本自动下载并处理一个小规模开源语料（如 WikiText-2 子集、莎士比亚文集或简单的中文小说文本）。
2. **分词器 (Tokenizer) 实验**：
   - 实现或调用两种不同的 Tokenizer 策略（如基础的 Char-level / Word-level 对比 BPE / WordPiece），生成不同大小的词表。
3. **模型训练**：
   - 训练 Transformer 拟合该语料，评估指标使用 Perplexity (PPL)。

---

# 第三阶段：实验验证与可视化 (Experiments & Visualization)

1. **消融实验与调参 (Ablation & Hyperparameter Tuning)**：
   - 测试并记录不同参数（如 `d_model`, `nhead`, `num_layers`, `dropout`）对两个子任务最终性能的影响。
   - 重点对比 Encoder-Decoder 架构与 Decoder-only 架构的差异。
2. **图表绘制**：
   - 使用 `matplotlib` 或 `seaborn` 绘制训练/验证 Loss 曲线。
   - 绘制加法任务中不同长度组合（如 3+3 vs 5+3）的准确率柱状图。
   - 绘制 Attention Map 热力图（可视化某一个加法等式生成时的注意力权重，特别是源序列和目标序列的对齐情况）。

# 交付要求
请按以下顺序逐步输出：
1. 先给出整体项目的文件结构树设计（推荐 `models/`, `data/`, `experiments/`, `utils/` 的结构）。
2. 提供核心的 Transformer 架构代码。
3. 提供数据生成与预处理代码。
4. 提供训练 Pipeline 和评估绘图代码。
请分步骤执行，每完成一个模块后等待我的确认或提问。