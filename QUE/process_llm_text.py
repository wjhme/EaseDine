import pandas as pd


df = pd.read_csv("/mnt/disk/wjh23/EaseDine/QUE/FunASR_A_audio_best_model_189_train_val_avg_llm_process.txt",sep="\t")

# print(df[df['text']!=df['llm_text']][30:80])
# df["llm_text"] = df["llm_text"].str.replace(r'[\s，。？！,.?!\d+]', '', regex=True)
# # 当text和llm_text长度不同时，用text替换llm_text
# for idx, row in df.iterrows():
#     text_len = len(str(row['text']))
#     llm_text_len = len(str(row['llm_text']))

#     if text_len != llm_text_len:
#         df.at[idx, 'llm_text'] = row['text']
df.drop('text',axis=1,inplace=True)
df.to_csv("/mnt/disk/wjh23/EaseDine/ASR/FunASR/A_audio_results/FunASR_A_audio_best_model_189_train_val_avg_llm_process.txt",sep="\t",index=False,header=None)