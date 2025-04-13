import pandas as pd


path = "/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/FireRed_all_batch_results/FireRed_train_audio_batch_1_beam_size_5.txt"
df = pd.read_csv(path, sep="\t")
select_df = df[~df['pred_text'].str.contains('天猫精灵', regex=True)]
text_exist = select_df[select_df['text'].str.contains('天猫精灵',regex=True)]
print(f"识别内容不含‘天猫精灵’ 的数量：{select_df.shape[0]}.\n其中，真实内容中包含的数量：{text_exist.shape[0]}.\n占比：{text_exist.shape[0]/select_df.shape[0]:.2f}")