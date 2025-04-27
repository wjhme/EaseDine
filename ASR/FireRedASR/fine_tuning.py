from fireredasr.models.fireredasr import FireRedAsr
from EaseDine.ASR.utils import load_data
from pathlib import Path
import time

# 当前脚本所在目录
CUR_DIR = Path(__file__).parent
ROOT_DIR = CUR_DIR.parent.parent.parent

# 配置路径
PRETRAINED_MODEL_PATH = f"{ROOT_DIR}/models/FireRedASR_pretrained_model/FireRedASR-AED-L"
DATA_DIR = f"{ROOT_DIR}/EaseDineDatasets"  # 包含音频和标注文件的目录
TRAIN_AUDIO_DIRS = f"{DATA_DIR}/train_audio"
TRANSCRIPTIONS_FILE = f"{DATA_DIR}/annotation.txt"

# 加载预训练模型
model = FireRedAsr.from_pretrained("aed", PRETRAINED_MODEL_PATH)

train_data = load_data(TRANSCRIPTIONS_FILE, TRAIN_AUDIO_DIRS)
print(f"数据集大小:{len(train_data)}")
# 微调配置
fine_tune_config = {
    "use_gpu": 1,                     # 是否使用 GPU
    "batch_size": 32,                 # 批量大小
    "learning_rate": 1e-4,            # 学习率
    "num_epochs": 10,                 # 训练轮数
    "save_dir": f"{ROOT_DIR}/models/FireRedASR_pretrained_model/fine_tuned_model",  # 保存微调后的模型路径
}

# 开始微调
print("开始微调模型...")
t1 = time.time()
model.fine_tune(
    train_data=train_data,
    config=fine_tune_config
)
print(f"微调用时:{(time.time() - t1)/60:.2f} 分")

print("微调完成！模型已保存至:", fine_tune_config["save_dir"])