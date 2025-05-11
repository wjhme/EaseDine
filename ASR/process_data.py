'''
处理 语音识别模型 生成的 uuid_hypothesis_cer 
按照 cer 筛选数据
生成 .scp 文件 格式：uuid audio_path
'''

import pandas as pd
from utils import get_files, merge_txt_files_to_dataframe

def select_data_by_cer_range(min_cer, max_cer, data_cer_dir):
    """
    选择 CER 在指定范围内的数据
    :param min_cer: CER 下限（包含）
    :param max_cer: CER 上限（包含）
    :param data_cer_dir: 数据目录路径
    :return: 筛选后的 DataFrame
    """
    # 获取目录下的文件列表
    count, filenames_ls = get_files(data_cer_dir)
    
    # 合并所有文件数据
    merge_df = merge_txt_files_to_dataframe(filenames_ls, data_cer_dir)
    
    # 筛选 CER 在 [min_cer, max_cer] 范围内的数据
    select_data = merge_df[
        (merge_df['cer'] >= min_cer) & 
        (merge_df['cer'] <= max_cer)
    ].reset_index(drop=True)
    
    # 打印统计信息和前10条样例
    print(f"数据集大小（{min_cer} ≤ cer ≤ {max_cer}）: {select_data.shape[0]}")
    print(select_data[['uuid', 'pred_text', 'text', 'cer']].head(10))
    
    return select_data


if __name__=="__main__":
    uuid_hypothesis_cer = "/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis_batchs_cer"
    
# # ================== 0.0 < cer < 0.3 的数据 ===============================
#     min_cer, max_cer = 0.0, 0.3
#     select_data = select_data_by_cer_range(min_cer, max_cer, data_cer_dir = uuid_hypothesis_cer)
#     select_data[['uuid','uuid_path']].to_csv("/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/train_val_03.scp",sep=" ",header=None,index=False)

# ================== 0.0 < cer < 0.1 的数据 ===============================
    min_cer, max_cer = 0.6, 2.0
    select_data = select_data_by_cer_range(min_cer, max_cer, data_cer_dir = uuid_hypothesis_cer)
    select_data[['uuid','pred_text','text','cer']].to_csv("/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/train_val_1_2.scp",sep="\t",index=False)
    # select_data[['uuid','uuid_path']].to_csv("/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/train_val_1_2.scp",sep=" ",header=None,index=False)


# # 划分数据集和验证集
# from sklearn.model_selection import train_test_split
# train_df, val_df = train_test_split(
#     cer_0[['uuid','uuid_path']],
#     test_size=0.15,       # 测试集比例
#     random_state=42      # 随机种子
# )
# train_df.to_csv("/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/train.scp",sep=" ",header=None,index=False)
# val_df.to_csv("/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/val.scp",sep=" ",header=None,index=False)
# cer_over_0[['uuid','uuid_path']].to_csv("/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/cer_over_0.scp",sep=" ",header=None,index=False)

# cer_over_0[['uuid','pred_text',"text",'cer']].to_csv("/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/cer_over_0.txt",sep="\t",index=False)
