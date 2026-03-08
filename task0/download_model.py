from huggingface_hub import snapshot_download
import os

# 配置参数（按需修改）
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 你选的模型名
SAVE_DIR = "/home/bravebeter/models/qwen2.5-1.5b"  # 模型保存目录（替换为你的工作目录）

# 创建保存目录（如果不存在）
os.makedirs(SAVE_DIR, exist_ok=True)

# 开始下载（断点续传，支持中断后重新运行）
print(f"开始下载模型 {MODEL_NAME} 到 {SAVE_DIR}...")
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=SAVE_DIR,
    resume_download=True,  # 断点续传
    local_dir_use_symlinks=False,  # 避免WSL符号链接问题
    max_workers=4  # 多线程下载，加快速度
)
print("模型下载完成！")