'''
读取测试数据集 -> 使用语音识别模型生成text -> 使用deepseek-v3进行纠错 -> 使用分类模型对text进行分类 -> 输出包含text和dom的结果文件.
'''

import torch
import json
from utils import llm_corrector
from ASR.FunASR.funasr import ASR

import os
import sys
import time
import pandas as pd
from DOM.Stacking.Classifiers import DOM


# ================================ 语音识别模块 ===================================
t0 = time.time()
# 启动前预加载模型（可选）
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 模型路径
model_cache_dir = "/mnt/disk/wjh23/models/FunASR_finetuner_models/189_mode_raw_enhanced_epcho_48"
asr = ASR(model_cache_dir=model_cache_dir)

# 音频文件/目录路径
input_path = "/mnt/disk/wjh23/EaseDineDatasets/dataset_b"  
output_file = "/mnt/disk/wjh23/EaseDine/ASR/FunASR/dataset_b/B_audio_189_mode_raw_enhanced_epoch_48.txt"
# input_path = "/mnt/disk/wjh23/EaseDineDatasets/train_audio/train_audio_batch_1"
# output_file="/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis/batch_1_enhence_model_allbatch_0_1_best.txt"

# 进行语音识别
asr.batch_process(
    input_path=input_path,
    output_file = output_file
)

print(f"语音识别耗时:{(time.time() - t0)/60:.2f}分钟.")

# # 官方提交文档
# ref_df = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label_B/B.txt",sep="\t")[['uuid']]
# # 生成结果文件
# data_df = pd.read_csv(output_file,sep="\t",header=None,names=['uuid','text'])
# # 处理uuid顺序
# data_df = ref_df.merge(data_df, on='uuid', how='left')
# # 保存处理后结果
# data_df.to_csv(output_file, sep="\t", index=False,header=None)

# # ================================ 大模型纠错模块 ===================================
# t0 = time.time()
# # 语音识别模型输出文件（uuid text）
# data_path = output_file
# # 大模型纠错结果保存路径（uuid text）
# save_corrected_path = "/mnt/disk/wjh23/EaseDine/ASR/FunASR/dataset_b/B_audio_189_mode_raw_enhanced_epoch_12_llm.txt"
# # 大模型纠错后与原文本存在差异数据记录保存路径（uuid text llm_text）
# save_diff_path = "/mnt/disk/wjh23/EaseDine/ASR/FunASR/dataset_b/diff/B_audio_189_mode_raw_enhanced_epoch_12_llm_diff.txt"

# #　进行大模型纠错
# llm_corrector(data_path, save_corrected_path, save_diff_path)

# print(f"大模型纠错耗时:{(time.time() - t0)/60:.2f}分钟.")


# ================================ 意图识别模块 ===================================
t0 = time.time()
# 语音识别文本数据路径
recognized_path = output_file
#　意图识别输出路径
save_path = "/mnt/disk/wjh23/EaseDine/DOM/dataset_b/B_audio_dom_189_mode_raw_enhanced_epoch_48.txt"
dom = DOM()
dom.pre_dom(recognized_path, save_path)

print(f"意图识别耗时:{(time.time() - t0)/60:.2f}分钟.")


# ================================ 菜品推荐模块 ===================================
# t0 = time.time()




# ================================ 处理uuid提交顺序 ===================================
# 官方提交文档
ref_df = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label_B/B.txt",sep="\t")[['uuid']]
# 生成结果文件
data_df = pd.read_csv(save_path,sep="\t")
# 处理uuid顺序
data_df = ref_df.merge(data_df, on='uuid', how='left')
# 保存处理后结果
data_df.to_csv(save_path, sep="\t", index=False)