import torch
import json
import pandas as pd
from FunASR import ASR

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

model_cache_dir = "/mnt/disk/wjh23/models/FunASR_finetuner_models/batch_enhence_model_allbatch_0_1_best"
asr = ASR(model_cache_dir=model_cache_dir)

# ================= 单文件/批处理 ============================
input_path = "/mnt/disk/wjh23/EaseDineDatasets/A_audio"  # 替换为你的音频文件/目录路径
# input_path = "/mnt/disk/wjh23/EaseDineDatasets/train_audio/train_audio_batch_1"
asr.batch_process(
    input_path=input_path,
    # output_file="/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis/batch_1_enhence_model_allbatch_0_1_best.txt",
    output_file="/mnt/disk/wjh23/EaseDine/ASR/FunASR/A_audio_results/FunASR_A_audio_best_model_allbatch_0_1_best.txt",
    sort_results = True
)

# result = asr.process_file("/mnt/disk/wjh23/EaseDineDatasets/train_audio/train_audio_batch_3/407683a6-905c-47cb-ac99-fc4fe90308b2.wav")
# print(result)

# # =================== 方言处理 ===========================
# fangyan = "/mnt/disk/wjh23/Test/filtered_results.txt"
# fangyan_df = pd.read_csv(fangyan, sep="\t")
# uuid_ls = fangyan_df['uuid'].tolist()

# uuid_dict_path = "/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/audio_paths.json"
# with open(uuid_dict_path, 'r', encoding='utf-8') as f:
#     uuid_dict = json.load(f)

# audio_files = []
# for audio in uuid_ls:
#     audio_files.append(uuid_dict[audio])

# asr.batch_process(
#     audio_files, 
#     output_file="/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis/fangyan_finetuner_best_model_189_train_val.txt", 
#     sort_results=False
# )
