import time
import torch
from vllm import LLM, SamplingParams

# ====================== 控制变量（与vLLM启动参数一致） ======================
MODEL_PATH = "/home/bravebeter/models/qwen2.5-1.5b"
MAX_MODEL_LEN = 2048
MAX_TOKENS = 50  # vLLM对应参数是max_tokens，而非max_new_tokens
TEMPERATURE = 0.7
TOP_P = 0.95
DTYPE = "float16"

# Qwen标准prompt格式（与API测试一致）
PROMPT = "<|im_start|>user\n你好，请用100字以内介绍人工智能的应用场景<|im_end|>\n<|im_start|>assistant\n"

def main():
    # 初始化vLLM引擎（严格匹配显存限制）
    llm = LLM(
        model=MODEL_PATH,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.75,  # 与启动参数一致
        max_num_batched_tokens=256,    # 与启动参数一致
        max_num_seqs=2,                # 与启动参数一致
        trust_remote_code=True
    )
    
    # 采样参数（修正max_tokens，与Transformers的max_new_tokens等价）
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,         # 核心修正：替换max_new_tokens为max_tokens
        temperature=TEMPERATURE,
        top_p=TOP_P,
        skip_special_tokens=True
    )

    # 预热（避免首次加载模型的耗时干扰）
    print("===== 预热模型 =====")
    llm.generate([PROMPT], sampling_params)

    # 正式测试（统计耗时）
    print("\n===== 开始vLLM推理测试 =====")
    start_time = time.time()
    # 单条请求推理
    outputs = llm.generate([PROMPT], sampling_params)
    end_time = time.time()

    # 计算核心指标
    total_time = end_time - start_time
    generated_tokens = len(outputs[0].outputs[0].token_ids)
    tokens_per_second = generated_tokens / total_time

    # 输出结果
    print(f"输入Prompt: {PROMPT[:50]}...")
    print(f"生成文本: {outputs[0].outputs[0].text}")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"生成Token数: {generated_tokens}")
    print(f"推理速度: {tokens_per_second:.4f} tokens/秒")

    # 查看显存占用（可选，需安装psutil）
    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"峰值显存占用: {mem_used:.2f} GB")

if __name__ == "__main__":
    main()