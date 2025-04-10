import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from utils import load_models, get_vectorizers, generate_meta_features

# 初始化路径
cur_dir = Path(__file__).parent
root_dir = cur_dir.parent.parent.parent

# 加载测试数据
test_file_path = f"{root_dir}/EaseDineDatasets/pred_A_audio.txt"
data = pd.read_csv(test_file_path, sep="\t", header=None, names=["uuid", "text"])

# 提取测试文本
X_test = data["text"].tolist()


# 加载模型
base_models, meta_model = load_models(root_dir)
print(f"成功加载 {len(base_models)} 个基模型")

# 生成元特征
X_meta = generate_meta_features(base_models, data["text"].tolist())

# 元模型预测
predictions = meta_model.predict(X_meta)

# 将结果添加到原始数据中
data['dom'] = predictions

data.to_csv(f"{root_dir}/EaseDineDatasets/pred_A_audio_with_dom.txt", sep="\t", header=None, index=None)

# 打印前几条预测结果
print(data.head())