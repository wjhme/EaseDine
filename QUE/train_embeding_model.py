import sys
from pathlib import Path

# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录（假设项目根目录是 EaseDine 的父目录）
project_root = current_file.parent.parent  # 根据实际层级调整
# DOM_path = current_file.parent.parent
# 将项目根目录添加到 Python 路径
sys.path.append(str(project_root))
from DOM.utils import process_data, Embeding
import pandas as pd
import json

# train_data = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt",sep="\t")
# train_data = train_data[train_data['dom']==1]
# caipin_data = pd.read_csv("/mnt/disk/wjh23/EaseDine/QUE/food_filter.csv")
# combine_data = pd.concat([train_data, caipin_data], axis=0)
# processed_data = process_data(combine_data, stop_words="/mnt/disk/wjh23/EaseDine/DOM/stopwords/que_stopwords.txt")
# # print(processed_data[140:200])


em = Embeding()
# 训练Word3Vec模型
DIM = 300
# train_em.train_word2vec(processed_data,DIM,sg=1)

# 生成菜品向量
caipin_data = pd.read_csv("/mnt/disk/wjh23/EaseDine/QUE/food_filter.csv")
caipin_data = process_data(caipin_data, stop_words="/mnt/disk/wjh23/EaseDine/DOM/stopwords/que_stopwords.txt",drop_duplicates=False)

caipin_embeding = em.get_word2vec(caipin_data,f"/mnt/disk/wjh23/EaseDine/DOM/embeding_models/que_word2vec/word2vec_model_1_{DIM}.bin")
caipin_data['cls'] = [json.dumps(vec.tolist()) for vec in caipin_embeding]
print(caipin_data.shape)
print(caipin_data[:3])
caipin_data.to_csv("food_embeding.txt",index=False)
# 生成测试文本向量
