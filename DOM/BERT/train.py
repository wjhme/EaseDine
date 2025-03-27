from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import pandas as pd
import time

# === 1. 数据预处理 ===
# 数据集加载（示例使用TREC06格式）
df = pd.read_csv('E:/githubworkspace/EaseDine/DOM/rawData/data.csv')

# 数据集划分
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# print(sum(train_df['label']==1)/train_df.shape[0])
# print(sum(val_df['label'] == 1)/val_df.shape[0])

# 文本编码
from utils import encode_texts
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
max_length = 128  # BERT最大序列长度
train_encodings = encode_texts(tokenizer, train_df['email'], max_length)
val_encodings = encode_texts(tokenizer, val_df['email'], max_length)

# === 3. 创建Dataset ===
from utils import SpamDataset
train_dataset = SpamDataset(train_encodings, train_df['label'].values)
val_dataset = SpamDataset(val_encodings, val_df['label'].values)

# === 4. 模型定义 ===
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2  # 二分类任务
)

# === 5. 训练配置 ===
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=2,              # 训练轮数
    per_device_train_batch_size=16,  # 每个设备的训练批次大小
    per_device_eval_batch_size=32,   # 每个设备的评估批次大小
    warmup_steps=500,                # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    eval_strategy="epoch",     # 每轮结束进行评估
    save_strategy="epoch",           # 每轮结束保存模型
    load_best_model_at_end=True,     # 加载最佳模型
    metric_for_best_model='f1',      # 使用F1分数作为最佳模型的评判标准
    fp16=True                        # 启用FP16混合精度
)

# === 6. 训练与评估 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

try:
    t0 = time.time()
    trainer.train()
    print(f"训练时间：{(time.time()-t0)/60:.2f} min")
except Exception as e:
    print(f"训练过程中发生错误: {e}")

# 保存最终模型
model.save_pretrained('./final_model')
tokenizer.save_pretrained('./final_model')

# 检查输出目录是否存在 pytorch_model.bin
import os
if os.path.exists(os.path.join(training_args.output_dir, 'pytorch_model.bin')):
    print("模型已成功保存到:", training_args.output_dir)
else:
    print("未找到 pytorch_model.bin 文件，请检查训练过程和输出目录权限。")