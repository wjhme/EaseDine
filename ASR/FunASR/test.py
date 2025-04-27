from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh",
                  model_revision="v2.0.4",
                  vad_model="fsmn-vad", punc_model="ct-punc",disable_update=True 
                  # spk_model="cam++"
                  )
audio = ["/mnt/disk/wjh23/EaseDineDatasets/热词音频/30990015-cf8c-476e-98bc-c069d7df0d61.wav",
         "/mnt/disk/wjh23/EaseDineDatasets/热词音频/0a0c93c1-2b5c-42dc-aefa-6e238aad5051.wav",
         "/mnt/disk/wjh23/EaseDineDatasets/热词音频/8c98447d-8e68-46aa-aee6-4af98d76e1fe.wav",
         "/mnt/disk/wjh23/EaseDineDatasets/热词音频/80204fb1-9a86-4b20-8064-86878e848054.wav",
         "/mnt/disk/wjh23/EaseDineDatasets/热词音频/d9b70c2c-ae06-4ff0-82e8-677a6cc5fc83.wav",
         "/mnt/disk/wjh23/EaseDineDatasets/processed_audio/train_audio_batch_1/cleaned_0a5d2a8a-12f9-4ea5-89c3-9d3246db5dea.wav"
         ]
for  file in audio:
    res = model.generate(input=file, 
                batch_size_s=200, 
                hotword='/mnt/disk/wjh23/EaseDine/DataSets/hotword.txt')
    print(res[0]['text'])

# from funasr import AutoModel
# from funasr.utils.postprocess_utils import rich_transcription_postprocess

# model_dir = "iic/SenseVoiceSmall"

# model = AutoModel(
#     model=model_dir,
#     vad_model="fsmn-vad",
#     vad_kwargs={"max_single_segment_time": 30000},
#     device="cuda:0",
#     disable_update=True
# )

# # en
# res = model.generate(
#     input=f"/mnt/disk/wjh23/EaseDineDatasets/热词音频/8e428ec7-eb85-43c0-962d-8e62fe463d51.wav",
#     # cache={},
#     language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=False,
#     batch_size_s=60,
#     merge_vad=True,  #
#     merge_length_s=15,
#     hotword='安儿陈'
# )
# text = rich_transcription_postprocess(res[0]["text"])
# print(text)