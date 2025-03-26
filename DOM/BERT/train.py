# === 1. 数据预处理 ===
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# 数据集加载（示例使用TREC06格式）
df = pd.read_csv('E:/githubworkspace/EaseDine/DOM/rawData/data.csv')
# df['label'] = df['label'].map({'spam': 1, 'ham': 0})  # 标签编码

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
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  # 启用FP16混合精度
)

# === 6. 训练与评估 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

import time
t0 = time.time()
trainer.train()
print(f"训练时间：{(time.time()-t0)/60:.2f} min")

