from fireredasr.models.fireredasr import FireRedAsr
from utils import load_pred_data, save_results_to_txt
from pathlib import Path
import time
import torch

# 清理 GPU 缓存
torch.cuda.empty_cache()

# 设置 CUDA 内存分配策略
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 获取当前路径和根目录
CUR_DIR = Path(__file__).parent
ROOT_DIR = CUR_DIR.parent.parent.parent

# 配置路径
PRETRAINED_MODEL_PATH = f"{ROOT_DIR}/models/FireRedASR_pretrained_model/FireRedASR-AED-L"
DATA_DIR = f"{ROOT_DIR}/EaseDineDatasets"  # 包含音频和标注文件的目录
TRAIN_AUDIO_DIRS = f"{DATA_DIR}/A_audio"

# 准备数据
batch_uttid, batch_wav_path = load_pred_data(TRAIN_AUDIO_DIRS)

# 初始化模型
model = FireRedAsr.from_pretrained("aed", PRETRAINED_MODEL_PATH)

# 定义批量大小
BATCH_SIZE = 64  # 根据 GPU 内存调整批量大小

# 批量处理音频文件
all_results = []
total_elapsed = 0.0
t0 = time.time()

for i in range(0, len(batch_uttid), BATCH_SIZE):
    # 获取当前批次的数据
    sub_batch_uttid = batch_uttid[i:i + BATCH_SIZE]
    sub_batch_wav_path = batch_wav_path[i:i + BATCH_SIZE]

    # 调用模型进行推理
    results, elapsed = model.transcribe(
        sub_batch_uttid,
        sub_batch_wav_path,
        {
            "use_gpu": 1,
            "beam_size": 3,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.0,
            "aed_length_penalty": 0.0,
            "eos_penalty": 1.0
        }
    )

    # 累加结果和耗时
    all_results.extend(results)
    total_elapsed += elapsed

    # 打印当前批次的进度
    print(f"已处理 {i + len(sub_batch_uttid)} / {len(batch_uttid)} 个音频文件")

# 计算总耗时
total_time = time.time() - t0
print(f"\n所有音频文件处理完成！总用时：{total_time:.2f} s，推理用时：{total_elapsed:.2f} s")

# 保存结果到文件
save_results_to_txt(all_results, f"{DATA_DIR}/pred_A_audio.txt")