from utils import based_on_keywords
import pandas as pd
from pathlib import Path

# 初始化路径
cur_dir = Path(__file__).parent
root_dir = cur_dir.parent.parent.parent

# 加载测试数据  EaseDineDatasets/智慧养老_label/train.txt
file_path = f"{root_dir}/EaseDineDatasets/pred_A_audio.txt"
data = pd.read_csv(file_path, sep="\t", header=None, names=["uuid", "text"])
# file_path = f"{root_dir}/EaseDineDatasets/智慧养老_label/train.txt"
# data = pd.read_csv(file_path, sep="\t")

data = based_on_keywords(data)
data.to_csv(f"{root_dir}/EaseDineDatasets/A_audio_dom_model1.txt",sep="\t",index=False)
# print(data.head())
# print(data[data['dom']==0])
# print(data[data['dom']==-1])