'''使用FunASR模型识别所有训练集音频，用于预处理数据集，提供FunASR模型微调'''

from FunASR import ASR
import time
import torch

model_cache_dir = "/mnt/disk/wjh23/models/FunASR_finetuner_models/finetuner_enhanced_cer_over_0"
FILE_NAMES = ['train_audio_batch_1', 'train_audio_batch_3', 'train_audio_batch_7', 'train_audio_batch_8', 'train_audio_batch_2', 'train_audio_batch_4', 'train_audio_batch_9', 'train_audio_batch_6', 'train_audio_batch_5']
asr = ASR(model_cache_dir=model_cache_dir)

for file_name in FILE_NAMES:
    print("="*20 + f" 处理数据集:{file_name} " + "="*20)
    # 清理 GPU 缓存
    torch.cuda.empty_cache()

    # 设置 CUDA 内存分配策略
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    input_path = f"/mnt/disk/wjh23/EaseDineDatasets/train_audio/{file_name}"
    t0 = time.time()
    asr.batch_process(
        input_path=input_path,
        output_file=f"/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis_batchs/{file_name}.txt",
        sort_results = False
    )

    # 计算总耗时
    total_time = time.time() - t0
    print(f"\n所有音频文件处理完成！总用时：{total_time/60:.2f} min.")
    print("\n" + "="*60)