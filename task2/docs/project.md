# Task-2：基于深度学习的文本分类

## 项目结构
 
可参考：
task2/
|____docs/
|        |___project.md # 项目信息
|        |___record.md # 实验过程记录
|
|____raw_data/
|           |___new_test.tsv
|           |___new_train.tsv
|
|____temp_data/  # 存放预处理之后的中间数据
|
|____data_process.py # 数据预处理脚本
|
|____train.py # 模型训练脚本
|
|____analysis.py # 结果测试与图表可视化
|
|____models/ # 保存训练好的模型

## 技术栈

1. Python
2. PyTorch
3. CUDA（GPU 加速）
4. uv

## 准备阶段

在开始 Task-2 之前，需要完成以下准备工作：

1. 学习 PyTorch 的基本使用方法，包括：

   * Tensor 的基本操作
   * 自动求导机制
   * Dataset 与 DataLoader 的使用

2. 重点掌握以下内容：

   * `torch.nn.Embedding` 的使用方法
   * `torch.nn.Conv1d / Conv2d` 等 CNN 相关模块
   * 使用 CUDA 将模型和数据迁移到 GPU 上进行训练

3. Task-2 的实现**不要求从底层实现模型结构**，所有核心组件都直接调用 PyTorch 的高级 API：
   - `nn.Embedding` - 词嵌入层
   - `nn.Conv1d` - 一维卷积层
   - `nn.MaxPool1d` - 最大池化层
   - `nn.RNN` / `nn.LSTM` / `nn.GRU` - 循环神经网络层
   - `nn.TransformerEncoder` - Transformer编码器
   - `nn.Linear` - 全连接层
   - `nn.Dropout` - Dropout层

   **无需手动实现任何算法的底层逻辑**。

## 数据集介绍

1. 训练数据和测试数据同 Task-1
2. 训练数据共 8528 条，测试数据共 3309 条
3. 每条数据为一条完整影评
4. 标签范围为 0–4，共 5 个类别

## 代码实现

### 1. 数据预处理

`data_process.py`

对 `raw_data` 文件夹中的数据进行预处理，并完成文本到序列的转换。

主要步骤：

1. 读取 `new_train.tsv` 和 `new_test.tsv`
2. 对文本进行基本清洗和分词
3. 构建词表（vocabulary）
4. 将文本转换为 token 序列
5. 将序列转换为固定长度的 index 序列
6. 使用 `torch.nn.Embedding` 在模型训练阶段完成 embedding 操作

数据集划分方式与 Task-1 相同：

* 训练集
* 验证集
* 测试集

处理后的数据保存至 `temp_data` 文件夹。

### 2. 模型训练

`train.py`

在本任务中，使用深度学习模型完成文本分类。

#### （1）CNN 文本分类模型

使用 PyTorch 提供的 CNN 相关 API 构建模型，包括：

* Embedding Layer
* Convolution Layer
* Pooling Layer
* Fully Connected Layer
* Softmax 输出层

模型结构示例：

Input → Embedding → CNN → MaxPooling → Fully Connected → Softmax

训练过程中需要：

* 使用 GPU（CUDA）进行加速
* 使用 DataLoader 进行 batch 训练
* 输出每个 epoch 的：

  * Loss
  * 验证集准确率
  * 测试集准确率

训练完成后，将模型保存到 `models` 文件夹中。

### 3. 实验设计

在本任务中，需要通过实验分析不同模型和参数对文本分类性能的影响。

实验内容包括：

#### （1）不同训练参数的影响

测试以下因素：

* 不同损失函数
* 不同学习率

分析其对最终分类性能的影响。

#### （2）CNN 结构参数的影响

测试不同 CNN 结构参数，包括：

* 卷积核数量
* 卷积核大小
* 不同优化器（如 SGD、Adam 等）

分析不同参数对模型性能的影响。

#### （3）预训练词向量

使用 GloVe 预训练词向量进行 embedding 初始化：

https://nlp.stanford.edu/projects/glove/

比较以下两种方式的性能差异：

* 随机初始化 embedding
* 使用 GloVe 初始化 embedding

分析预训练词向量对模型效果的影响。

#### （4）不同模型结构的比较

在 CNN 模型的基础上，尝试使用其他深度学习模型进行文本分类：

* RNN
* LSTM / GRU
* Transformer

上述模型可以直接调用 PyTorch 提供的 API 实现。

比较不同模型结构在该任务上的性能差异。

### 4. 结果分析

`analysis.py`

对实验结果进行分析，并进行可视化展示。

主要内容包括：

1. 比较不同模型结构（CNN、RNN、Transformer）的性能
2. 比较不同 CNN 参数配置的性能
3. 比较是否使用预训练词向量的差异
4. 比较不同学习率和优化器的影响

最终结果需要：

* 统计模型在测试集上的分类准确率
* 绘制实验结果图表（如折线图、柱状图等）
* 对实验结果进行总结分析
