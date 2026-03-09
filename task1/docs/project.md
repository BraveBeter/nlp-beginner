你需要实现一个完整的基于机器学习的文本分类任务。以下是详细要求：

项目结构：
- task1/
    - docs/
        - project.md    # 项目信息
        - record.md     # 实验记录
    - raw_data/
        - new_train.tsv
        - new_test.tsv
    - temp_data/       # 中间处理数据
    - data_process.py  # 数据预处理脚本
    - train.py         # 模型训练脚本
    - analysis.py      # 结果测试与图表可视化
    - models/          # 保存训练好的模型

任务步骤：

1. 数据预处理 (`data_process.py`)：
    - 输入：raw_data 文件夹中的训练和测试数据
    - 输出：可直接训练的向量化数据，保存到 temp_data
    - 方法：
        - 使用 Bag of Words
        - 使用 N-gram (N<=3)
    - 注意：每种方法分别生成独立的向量化数据

2. 模型训练 (`train.py`)：
    - 模型：
        - softmax 多分类
        - 感知机多分类
    - 数据：
        - Bag of Words 和 N-gram
    - 损失函数：
        - softmax -> 交叉熵损失
        - 感知机 -> 感知机损失
    - 训练要求：
        - 不可直接调用 torch.nn 的高阶函数
        - 使用 PyTorch 矩阵操作对整个 batch 并行处理
        - 可以调整学习率和损失函数做组合训练
    - 输出：
        - 保存训练好的模型到 models/
        - 每轮输出 loss 和验证集准确率

3. 结果分析 (`analysis.py`)：
    - 对比 Bag of Words 与 N-gram 的性能
    - 对比不同损失函数和学习率的影响
    - 绘制图表展示结果
    - 输出测试集的分类准确率和混淆矩阵

额外要求：
- 代码风格清晰、可直接运行
- 使用 Python 和 PyTorch
- 所有输出保存到对应文件夹，并可复现实验过程

数据集说明：
- 训练数据：8528 条
- 测试数据：3309 条
- 标签：0~4（5 类）

请基于以上描述生成完整可运行的代码，实现数据预处理、模型训练与结果分析。