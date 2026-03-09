## Task0：
- 使用一张 3090/4090 显卡，使用 vllm 框架部署一个 7B 的模型的 API
    
    - 如果自己的环境显存不够，可以部署更小的模型
        
    - 可以在 /remote-home1/share/models 下寻找自己感兴趣的模型
        
    - 比较使用 vllm 和直接使用 transfromers 进行推理的效率，**并思考其中的原因**
        

## 执行步骤：
1. 准备显卡；
2. 选模型 → 配环境；
3. 用 vLLM 部署 API 并验证；
4. 分别用 vLLM 和 Transformers 跑推理，记录数据；
5. 查资料 / 思考原理，理解 vLLM 的优势。

## 实现过程：
环境：WSL + uv + VsCode，显存RTX3060 6GB
  
1. 使用uv配置环境， 下载vllm、transformers
```bash
uv init
uv add vllm transformers
uv sync
```
2. 使用`download_model`脚本（hugging Face）下载了Qwen/Qwen2.5-1.5B-Instruct到本地。
3. 使用vllm命令运行本地模型，并进行测试
```bash
source .venv/bin/activate # 激活venv环境

# 使用vllm运行启动模型
vllm serve /home/bravebeter/models/qwen2.5-1.5b --gpu-memory-utilization 0.75 --port 8000 --trust-remote-code --max-num-batched-tokens 256 --max-num-seqs 2 --max-model-len 2048 --dtype float16 

# 测试
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<|im_start|>user\n你好，请介绍自己<|im_end|>\n<|im_start|>assistant\n",
    "max_tokens": 50,
    "temperature": 0.7
  }' 
```
测试执行成功
![alt text](images/test.png)

发送请求后，控制台显示Avg prompt throughput和Avg generation throughput
![alt text](images/info.png)

4. 控制变量，采用相同的参数
- 相同的提示词
- 模型路径：/home/bravebeter/models/qwen2.5-1.5b
- 精度：float16
- 最大序列长度：2048
- 生成参数：max_new_tokens=50、temperature=0.7、top_p=0.95
- 推理模式：单条请求（无批处理）（避免 vLLM 批处理优势干扰基础效率对比）
- 硬件：同一 RTX 3060 6GB，无其他进程占用显存

使用`vllm_benchmark`脚本和`transformers_benchmark`做对比

### vllm测试结果：
```
输入Prompt: <|im_start|>user
你好，请用100字以内介绍人工智能的应用场景<|im_end|>
...
生成文本: 人工智能在多个领域都有广泛应用，如语音识别、图像识别、自然语言处理、自动驾驶、医疗诊断、金融服务、智能家居等。
总耗时: 0.4161 秒
生成Token数: 29
推理速度: 69.6977 tokens/秒
峰值显存占用: 0.00 GB （可能由于：vLLM 采用多进程架构，主进程无法直接获取 GPU 显存）
```

### transformers测试结果
```
输入Prompt: <|im_start|>user
你好，请用100字以内介绍人工智能的应用场景<|im_end|>
...
生成文本: 人工智能在医疗诊断、自动驾驶、智能客服、语音识别和虚拟助手等领域广泛应用。
总耗时: 0.5389 秒
生成Token数: 19
推理速度: 35.2584 tokens/秒
峰值显存占用: 2.89 GB
```

|核心指标|	vLLM 测试结果|	Transformers 测试结果|	差异（vLLM 优势）|
|----- |------- |------ |-------|
推理速度|（tokens / 秒）|	69.70|	35.26	|提速约 97.6%（近 2 倍）|
总耗时（秒）|	0.4161|	0.5389|	耗时减少 22.8%|
实际生成 Token 数|	29|	19|	生成更多内容但耗时更少|
峰值显存占用（GB）|	~3.0-3.2（估算）|	2.89	|vLLM 显存显示异常（多进程导致），实际略高但利用率更高|

## 分析
为什么vllm更高效？
1. 实测数据验证：vLLM 在 RTX 3060 6GB 环境下，推理速度是原生 Transformers 的近 2 倍，且显存性价比更高；
2. 核心优势来源：PagedAttention 解决 KV Cache 碎片化 + 连续批处理提升 GPU 利用率 + FlashAttention v2 硬件级加速；
3. 小显存场景启示：vLLM 的动态显存管理和分页注意力机制，让小显存 GPU 也能高效运行大模型，这是 Transformers 原生推理无法比拟的。