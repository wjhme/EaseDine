'''
读取数据集 -> 使用语音识别模型生成text -> 使用分类模型对text进行分类 -> 输出包含text和dom的结果文件.
'''
import torch
import pandas as pd
from ASR.my_funasr import ASR
from DOM.Stacking.Classifiers import DOM 
from QUE.llm_recommend import recommend

# ================================ 语音识别模块 =================================== 
# 启动前预加载模型
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 语音识别模型路径 https://www.modelscope.cn/models/wjh6002/speech_recognition_moel
model_cache_dir = "speech_recognition_moel"
asr = ASR(model_cache_dir=model_cache_dir)

# 音频文件/目录路径
input_path = "./datasets_test"  # 替换为真实数据集路径
output_file="results/speech_recognition.txt"

# 进行语音识别
asr.batch_process(
    input_path=input_path,
    output_file = output_file
)

# ================================ 意图识别模块 ===================================
# 语音识别文本数据路径
recognized_path = output_file
#　意图识别输出路径
save_path = "results/dom.txt"
dom = DOM()
dom.pre_dom(recognized_path, save_path)

# ================================ 菜品推荐模块 ===================================
data_path = save_path
save_path = "results/Result.txt"
recommend(data_path, save_path)

# # ================================ 处理uuid提交顺序 ===================================
# # 官方提交文档
# ref_df = pd.read_csv("project/B.txt",sep="\t")[['uuid']]
# # 生成结果文件
# data_df = pd.read_csv(save_path,sep="\t")
# # 处理uuid顺序
# data_df = ref_df.merge(data_df, on='uuid', how='left')
# # 保存处理后结果
# data_df.to_csv(save_path, sep="\t", index=False)