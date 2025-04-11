# import torch
# from FunASR import main_process
#
#
# # 启动前预加载模型（可选）
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#
# # 测试文件
# audio_file = "../audios/test1.wav"
# main_process(audio_file)

from FunASRProcessor import FunASRProcessor
from FunASRFineTuner import FunASRFineTuner

# 初始化处理器
processor = FunASRProcessor()

# 单文件识别
result = processor.transcribe("test.wav")
print(f"识别结果: {result['text']}")

# 批量处理
processor.batch_process("audio_directory/", "results.tsv")

# 模型微调
fine_tuner = FunASRFineTuner(
    model_name="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    train_data="path/to/train_data",
    valid_data="path/to/valid_data",
    num_epochs=15
)
fine_tuner.train()
fine_tuner.save_model("custom_model/")