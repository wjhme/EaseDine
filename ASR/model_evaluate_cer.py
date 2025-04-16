import pandas as pd
from FunASR.cer_evaluate import calculate_cer

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

if __name__ == "__main__":
    # ========================== FireRedASR =======================================
    # nohup python model_evaluate_cer.py > FireRedASR/log/cer_out.log 2>&1 &

    # FILE_NAMES = ['train_audio_batch_1', 'train_audio_batch_3', 'train_audio_batch_7', 'train_audio_batch_8', 'train_audio_batch_2', 'train_audio_batch_4', 'train_audio_batch_9', 'train_audio_batch_6', 'train_audio_batch_5']
    # for file_name in FILE_NAMES:
    #     hypothesis = f"/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/uuid_hypothesis/FireRed_{file_name}.txt"
    #     reference = "/mnt/disk/wjh23/EaseDine/DOM/train.txt"
    #     save = f"/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/uuid_hypothesis_cer/{file_name}.txt"

    #     ans = CER(reference, hypothesis, save)
    #     print(f"{file_name} 数据集 CER 得分:{ans:.4f}\n")

    
    hypothesis = f"/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis/fangyan_finetuner_large.txt"
    reference = "/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt"
    # reference = "/mnt/disk/wjh23/EaseDine/DOM/train.txt"
    save = f"/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_all_batch_results/uuid_hypothesis_cer/fangyan_finetuner_large_cer.txt"

    ans = CER(reference, hypothesis, save)
    print(f"audio_batch_1_beam_size_5 数据集 CER 得分:{ans:.4f}\n")

    # # ================================== FireRedASR ================================================

    # hypothesis = pd.read_csv(r"E:\githubworkspace\EaseDine\ASR\FireRedASR\FireRed_train_audio_batch_1.txt",sep="\t") # time: min-0.06;max-6.37;mean-0.07
    # hypothesis.columns = ['uuid','pred_text']
    # print(hypothesis[hypothesis['pred_text'].isna()])
    # hypothesis['pred_text'] = hypothesis['pred_text'].fillna("")
    # hypothesis['pred_text'] = hypothesis['pred_text'].str.replace(r'[a-zA-Z\s，。？！]', '', regex=True)

    # reference = pd.read_csv(r"E:\githubworkspace\EaseDine\DOM\train.txt",sep="\t")
    # # 删除字母和空格
    # reference['text'] = reference['text'].str.replace(r'[a-zA-Z\s，。？！]', '', regex=True)

    # # print(reference.head())
    # # print(hypothesis.head())

    # result_df = hypothesis.merge(reference[['uuid', 'text']], on='uuid', how='left')[['text','pred_text']]

    # # print(result_df.head(10))
    # text_pairs = list(zip(result_df['text'], result_df['pred_text']))

    # fire_red_asr_cer = []
    # for ref, hyp in text_pairs:
    #     # print(ref,hyp)
    #     result = calculate_cer(ref, hyp)
    #     fire_red_asr_cer.append(result)

    # print(f"FireRedASR train_audio_batch_1 ASR得分:{1 - sum(fire_red_asr_cer)/len(fire_red_asr_cer):.4f}")