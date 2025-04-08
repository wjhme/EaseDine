from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from pathlib import Path
import time

t0 = time.time()
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('/mnt/disk/wjh23/models/bert_model/final_model')
# 加载模型
model = AutoModelForSequenceClassification.from_pretrained('/mnt/disk/wjh23/models/bert_model/final_model')
print(f"加载模型耗时：{time.time() - t0}s.")

# 初始化路径
cur_dir = Path(__file__).parent
root_dir = cur_dir.parent.parent.parent

# 加载测试数据
test_file_path = f"{root_dir}/EaseDineDatasets/pred_A_audio.txt"
data = pd.read_csv(test_file_path, sep="\t", header=None, names=["uttid", "text"])

text = data['text'].tolist()
t1 = time.time()
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
# 进行预测
outputs = model(**inputs)
logits = outputs.logits
predictions = logits.argmax(dim=-1)
print(f"预测耗时：{time.time() - t1}s.")
# print("预测输出：",predictions.cpu(),"[1:正常邮件；0：垃圾邮件]")

'''
加载模型耗时：1.0578579902648926s.
预测耗时：0.2296619415283203s.
预测输出： tensor([1, 0, 1, 0]) [1:正常邮件；0：垃圾邮件]
'''
