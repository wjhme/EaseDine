import torch
import json
import pandas as pd
from FunASR.my_funasr import ASR

'''
nohup python main.py > log/batch_1_enhence_model_allbatch_0_1_best.log 2>&1 &
nohup python main.py > log/fangyan_finetuner_best_model_189_train_val.log 2>&1 &
nohup python main.py > log/batch_1_finetuner_4_28_cer_03.log 2>&1 &
nohup python main.py > log/FunASR_A_audio_best_model_allbatch_0_1_best.log 2>&1 &

# 模型：
finetuner_enhanced_cer_over_0
finetuner_4_28_cer_03
finetuner_model_234567_89_cer_06
batch_enhence_model_CMDS_part_data
batch_enhence_model_001_1
batch_enhence_model_189_train_val(avg)
batch_enhence_model_allbatch_0_1_best
'''

# 启动前预加载模型（可选）
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model_cache_dir = "/mnt/disk/wjh23/models/FunASR_finetuner_models/189_mode_raw_enhanced_epcho_second"
asr = ASR(model_cache_dir=model_cache_dir)

# ================= 单文件/批处理 ============================
# input_path = "/mnt/disk/wjh23/EaseDineDatasets/A_audio"  # 替换为你的音频文件/目录路径
# input_path = "/mnt/disk/wjh23/EaseDineDatasets/train_audio/train_audio_batch_1"
# output_file="/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis/batch_1_189_mode_raw_enhanced_epcho_43.txt"
# asr.batch_process(
#     input_path=input_path,
#     output_file = output_file
#     # output_file="/mnt/disk/wjh23/EaseDine/ASR/FunASR/A_audio_results/FunASR_A_audio_best_model_allbatch_0_1_best.txt"
# )

wav = "dc725a2e-2911-4b1a-8b83-73ce7e955a6e"
result = asr.process_file(f"/mnt/disk/wjh23/EaseDineDatasets/dataset_b/{wav}.wav")
# result = asr.process_file(f"/mnt/disk/wjh23/EaseDine/ASR/data_process/temp_split_audio/2.wav")
print(result)
