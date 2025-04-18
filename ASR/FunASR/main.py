import torch
from FunASR import main_process


# 启动前预加载模型（可选）
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 测试文件
audio_uuid = "0abed3b5-691d-4b35-b2b4-900a2f1777f2"
# audio_file = f"/mnt/disk/wjh23/EaseDineDatasets/存在噪音的语音/{audio_uuid}.wav"
audio_file = "/mnt/disk/wjh23/EaseDineDatasets/A_audio/0612d9d1-950e-4c8d-9c72-ac4a02f3481d.wav"
main_process(audio_file)
# process_file = f"/mnt/disk/wjh23/EaseDineDatasets/音频降噪后语音/{audio_uuid}_processed.wav"
# process_file = "/mnt/disk/wjh23/separated_audio/4d1826a9-2821-48e9-af58-d3136b796d71/vocals.wav"
# main_process(process_file)

# from FunASRProcessor import FunASRProcessor
# from FunASRFineTuner import FunASRFineTuner

# # 初始化处理器
# processor = FunASRProcessor()

# # 单文件识别
# result = processor.transcribe("test.wav")
# print(f"识别结果: {result['text']}")

# 批量处理
# processor.batch_process("audio_directory/", "results.tsv")

# # 模型微调
# fine_tuner = FunASRFineTuner(
#     model_name="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
#     train_data="path/to/train_data",
#     valid_data="path/to/valid_data",
#     num_epochs=15
# )
# fine_tuner.train()
# fine_tuner.save_model("custom_model/")