import pandas as pd
import sys
from pathlib import Path
# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录（假设项目根目录是 EaseDine 的父目录）
ASR_path = current_file.parent.parent
# 将项目根目录添加到 Python 路径
sys.path.append(str(ASR_path))

from utils import calculate_cer

def CER(reference_path, hypothesis_path, save_path):
    '''计算语音识别CER得分，并保存每条语音的cer'''
    # 加载识别数据
    hypothesis = pd.read_csv(hypothesis_path, sep="\t", header=None, names=['uuid','pred_text']) 

    # 查看未识别的语音
    print(hypothesis[hypothesis['pred_text'].isna()])
    hypothesis['pred_text'] = hypothesis['pred_text'].fillna("天猫精灵")
    hypothesis['pred_text'] = hypothesis['pred_text'].str.replace(r'[a-zA-Z\s，。？！]', '', regex=True)

    # 加载参考文本
    reference = pd.read_csv(reference_path, sep="\t")
    # 删除标点符号和空格
    reference['text'] = reference['text'].str.replace(r'[\s，。？！]', '', regex=True)

    merge_df = hypothesis.merge(reference[['uuid', 'text']], on='uuid', how='left')
    result_df = merge_df[['text','pred_text']]

    # 计算数据集CER得分
    text_pairs = list(zip(result_df['text'], result_df['pred_text']))

    fun_asr_cer = []
    for ref, hyp in text_pairs:
        # print(ref,hyp)
        result = calculate_cer(ref, hyp)
        fun_asr_cer.append(round(result,4))

    merge_df['cer'] = fun_asr_cer
    merge_df.to_csv(save_path, sep="\t", index=False)
    print(f"cer==0 的音频数量：{sum(merge_df['cer']==0.0)}")

    return 1 - sum(fun_asr_cer)/len(fun_asr_cer)

def A_CER(reference_path, hypothesis_path, save_path):
    '''计算语音识别CER得分，并保存每条语音的cer'''
    # 加载识别数据
    hypothesis = pd.read_csv(hypothesis_path, sep="\t", header=None, names=['uuid','pred_text']) 

    # 查看未识别的语音
    print(hypothesis[hypothesis['pred_text'].isna()])
    hypothesis['pred_text'] = hypothesis['pred_text'].fillna("天猫精灵")
    hypothesis['pred_text'] = hypothesis['pred_text'].str.replace(r'[a-zA-Z\s，。？！]', '', regex=True)

    # 加载参考文本
    reference = pd.read_csv(reference_path, sep="\t", header=None, names=['uuid','ref_text'])
    # 删除标点符号和空格
    reference['ref_text'] = reference['ref_text'].str.replace(r'[\s，。？！]', '', regex=True)

    merge_df = hypothesis.merge(reference[['uuid', 'ref_text']], on='uuid', how='left')
    result_df = merge_df[['ref_text','pred_text']]

    # 计算数据集CER得分
    text_pairs = list(zip(result_df['ref_text'], result_df['pred_text']))

    fun_asr_cer = []
    for ref, hyp in text_pairs:
        result = calculate_cer(ref, hyp)
        fun_asr_cer.append(round(result,4))

    merge_df['cer'] = fun_asr_cer
    merge_df.to_csv(save_path, index=False,encoding='utf_8_sig')
    print(f"cer==0 的音频数量：{sum(merge_df['cer']==0.0)}")

    return 1 - sum(fun_asr_cer)/len(fun_asr_cer)

if __name__ == "__main__":
    # ========================== FireRedASR =======================================
    # nohup python model_evaluate_cer.py > FireRedASR/log/cer_out.log 2>&1 &

    # FILE_NAMES = ['train_audio_batch_1', 'train_audio_batch_3', 'train_audio_batch_7', 'train_audio_batch_8', 'train_audio_batch_2', 'train_audio_batch_4', 'train_audio_batch_9', 'train_audio_batch_6', 'train_audio_batch_5']
    # for file_name in FILE_NAMES:
    #     hypothesis = f"/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis_batchs/{file_name}.txt"
    #     reference = "/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt"
    #     save = f"/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis_batchs_cer/{file_name}_cer.txt"

    #     ans = CER(reference, hypothesis, save)
    #     print(f"{file_name} 数据集 CER 得分:{ans:.4f}\n")

# fangyan_finetuner_enhenced_over_0_best
    data_name = "batch_1_189_mode_raw_enhanced_epcho_43"
    hypothesis = f"/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis/{data_name}.txt"
    save = f"/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis_cer/{data_name}_cer.txt"

    reference = "/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt"
    ans = CER(reference, hypothesis, save)
    print(f"{data_name} 数据集 CER 得分:{ans:.4f}\n")

# # ===========================  A榜测试集对比分析 ======================================================
#     data_name = "FunASR_A_audio_best_model_189_train_val_avg"
#     hypothesis = f"/mnt/disk/wjh23/EaseDine/ASR/FunASR/A_audio_results/{data_name}.txt"
#     # A榜最佳结果
#     reference = "/mnt/disk/wjh23/EaseDine/ASR/FunASR/A_audio_results/FunASR_A_audio_4_28.txt"
#     save = "/mnt/disk/wjh23/EaseDine/ASR/FunASR/A_audio_results/A榜cer对比/4_28_best_model_189_train_val_avg.csv"
#     ans = A_CER(reference, hypothesis, save)
#     print(f"{data_name} 数据集 CER 得分:{ans:.4f}\n")