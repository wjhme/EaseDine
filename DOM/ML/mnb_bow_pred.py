import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# 初始化路径
cur_dir = Path(__file__).parent
root_dir = cur_dir.parent.parent.parent

# 加载测试数据
test_file_path = f"{root_dir}/EaseDineDatasets/pred_A_audio.txt"
data = pd.read_csv(test_file_path, sep="\t", header=None, names=["uuid", "text"])

# 提取测试文本
X_test = data["text"].tolist()

bow_vectorizer = joblib.load('model/bow_vectorizer.joblib')
bow_test_features = bow_vectorizer.transform(X_test)
model = joblib.load("/mnt/disk/wjh23/EaseDine/DOM/ML/model/mnb_bow.pkl")

pred = model.predict(bow_test_features)

# 将结果添加到原始数据中
data['dom'] = pred

data.to_csv(f"{root_dir}/EaseDineDatasets/A_audio_mnb_bow.txt", sep="\t", header=None, index=None)
