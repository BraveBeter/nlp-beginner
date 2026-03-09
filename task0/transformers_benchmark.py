import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====================== 完全相同的控制变量 ======================
MODEL_PATH = "/home/bravebeter/models/qwen2.5-1.5b"
MAX_MODEL_LEN = 2048
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
TOP_P = 0.95
DTYPE = torch.float16

PROMPT = "<|im_start|>user\n你好，请用100字以内介绍人工智能的应用场景<|im_end|>\n<|im_start|>assistant\n"

def main():
    # 设置CUDA显存限制（模拟vLLM的gpu_memory_utilization=0.75）
    torch.cuda.set_per_process_memory_fraction(0.75, device=0)

    # 加载tokenizer和模型（严格匹配精度）
    print("===== 加载模型 =====")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        device_map="auto",  # 自动分配到GPU
        max_memory={0: "4.5GB"}  # 与vLLM 0.75利用率一致（6*0.75=4.5GB）
    )
    model.eval()  # 推理模式，关闭dropout

    # 编码prompt
    inputs = tokenizer([PROMPT], return_tensors="pt").to("cuda")
    # 限制输入长度（与vLLM max_model_len一致）
    if inputs["input_ids"].shape[1] > MAX_MODEL_LEN:
        inputs["input_ids"] = inputs["input_ids"][:, :MAX_MODEL_LEN]
        inputs["attention_mask"] = inputs["attention_mask"][:, :MAX_MODEL_LEN]

    # 预热
    print("===== 预热模型 =====")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True
        )

    # 正式测试
    print("\n===== 开始Transformers推理测试 =====")
    start_time = time.time()
    with torch.no_grad():  # 禁用梯度计算，节省显存+提速
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            max_length=MAX_MODEL_LEN  # 与vLLM一致
        )
    end_time = time.time()

    # 计算核心指标
    total_time = end_time - start_time
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    generated_tokens = len(outputs[0]) - inputs["input_ids"].shape[1]
    tokens_per_second = generated_tokens / total_time

    # 输出结果
    print(f"输入Prompt: {PROMPT[:50]}...")
    print(f"生成文本: {generated_text}")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"生成Token数: {generated_tokens}")
    print(f"推理速度: {tokens_per_second:.4f} tokens/秒")

    # 峰值显存占用
    mem_used = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"峰值显存占用: {mem_used:.2f} GB")

if __name__ == "__main__":
    main()