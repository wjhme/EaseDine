import torch
from FunASR import main_process


# 启动前预加载模型（可选）
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 测试文件
# aduio_uuid = "0c705c6d-166b-4a95-8798-4cada2b10691"
audio_file = f"/mnt/disk/wjh23/EaseDineDatasets/train_audio/train_audio_batch_1/066461e2-4587-4809-aaf4-ea56d3d7c19f.wav"
main_process(audio_file)
# process_file = f"/mnt/disk/wjh23/EaseDineDatasets/处理后音频/{aduio_uuid}_processed.wav"
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