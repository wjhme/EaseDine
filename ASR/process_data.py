'''
处理FireRedASR生成的 uuid_hypothesis_cer 
按照cer==0.0筛选数据
生成 .scp 文件 格式：uuid audio_path
'''

import pandas as pd
from EaseDine.ASR.utils import get_files, merge_txt_files_to_dataframe

uuid_hypothesis_cer = "/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/uuid_hypothesis_cer"
count, filenames_ls = get_files(uuid_hypothesis_cer)

base_path = "/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/uuid_hypothesis_cer"
merge_df = merge_txt_files_to_dataframe(filenames_ls, base_path)

# cer == 0.0 的数据
cer_0 = merge_df[merge_df['cer'] == 0.0].reset_index(drop=True)
print(f"cer_0 size:{cer_0.shape[0]}")
print(cer_0[['uuid','uuid_path']].head(10))
cer_0[['uuid','uuid_path']].to_csv("/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/cer_0.scp",sep=" ",header=None,index=False)

# cer != 0.0 的数据
cer_over_0 = merge_df[merge_df['cer'] != 0.0].reset_index(drop=True)
print(f"\ncer_over_0 size:{cer_over_0.shape[0]}")
print(cer_over_0[['uuid','uuid_path']].head(10))

# # 划分数据集和验证集
# from sklearn.model_selection import train_test_split
# train_df, val_df = train_test_split(
#     cer_0[['uuid','uuid_path']],
#     test_size=0.15,       # 测试集比例
#     random_state=42      # 随机种子
# )
# train_df.to_csv("/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/train.scp",sep=" ",header=None,index=False)
# val_df.to_csv("/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/val.scp",sep=" ",header=None,index=False)
cer_over_0[['uuid','uuid_path']].to_csv("/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/cer_over_0.scp",sep=" ",header=None,index=False)
