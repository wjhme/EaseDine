import torch
from FunASR import main_process


# 启动前预加载模型（可选）
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 测试文件
audio_file = "../audios/test1.wav"
main_process(audio_file)