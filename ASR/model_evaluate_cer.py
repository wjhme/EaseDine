import pandas as pd
from FunASR.CER_evaluate import calculate_cer

hypothesis = pd.read_csv("/mnt/disk/wjh23/EaseDine/ASR/FunASR/FunASR_train_audio_batch_11.txt",sep="\t") # time: min-0.06;max-6.37;mean-0.07
hypothesis.columns = ['uuid','pred_text','time']
hypothesis['pred_text'].fillna("")
hypothesis['pred_text'] = hypothesis['pred_text'].str.replace(r'[a-zA-Z\s，。？！]', '', regex=True)

reference = pd.read_csv("/mnt/disk/wjh23/EaseDine/DOM/train.txt",sep="\t")
# 删除字母和空格
reference['text'] = reference['text'].str.replace(r'[a-zA-Z\s，。？！]', '', regex=True)

# print(reference.head())
# print(hypothesis.head())

result_df = hypothesis.merge(reference[['uuid', 'text']], on='uuid', how='left')[['text','pred_text']]

print(result_df.head(10))
text_pairs = list(zip(result_df['text'], result_df['pred_text']))

fun_asr_cer = []
for ref, hyp in text_pairs:
    print(ref,hyp)
    result = calculate_cer(ref, hyp)
    fun_asr_cer.append(result)

print(f"FunASR train_audio_batch_1 ASR得分:{1 - sum(fun_asr_cer)/len(fun_asr_cer):.4f}")