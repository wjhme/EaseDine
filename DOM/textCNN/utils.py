import collections
import jieba  # 中文分词
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random

def get_tokenized(data, stopwords_path = "E:\githubworkspace\EaseDine\DOM\stopwords\my_stopwords.txt"):
    # 复制数据避免修改原数据
    df = data.copy()

    # 加载停用词表
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])

    def tokenize(text):
        # 分词并过滤停用词/空字符/标点
        words = jieba.lcut(str(text))
        return [
            word for word in words
            if (word not in stopwords) and       # 去停用词
               (word.strip() != '') and          # 去空字符
               (not word.isspace()) and          # 去空白符
               (not all(c in '，。！？；：“”‘’（）【】…—' for c in word))  # 去中文标点
        ]

    df['tokenized'] = df['text'].apply(tokenize)
    # print(max(df['text'].apply(lambda x:len(x))))
    return df

class Vocabulary:
    '''词汇表类'''
    def __init__(self, counter, min_freq=1, reserved_tokens=None):
        self.itos = reserved_tokens.copy() if reserved_tokens else []
        self.token_to_idx = {token: i for i, token in enumerate(self.itos)}

        # 添加频率足够的词
        for token, count in counter.items():
            if count >= min_freq and token not in self.token_to_idx:
                self.itos.append(token)
                self.token_to_idx[token] = len(self.itos) - 1

    def __getitem__(self, token):
        return self.token_to_idx.get(token, 0)

    def __len__(self):
        """返回词汇表大小"""
        return len(self.itos)

    def to_indices(self, tokens):
        """将token列表转换为索引列表"""
        return [self[token] for token in tokens]

# 从分词后的数据构建词汇表（Vocabulary），包含低频词过滤和保留特殊符号（如 <pad>）
def get_vocab(tokenized_data):
    # 1. 分词处理
    # tokenized_data = get_tokenized(data)
    # 2. 统计词频
    counter = collections.Counter([tk for st in tokenized_data['tokenized'] for tk in st])
    # 3. 构建词汇表
    return Vocabulary(counter, min_freq = 1, reserved_tokens = ['<pad>','<unk>'])


def process_data(tokenized_data, vocab, max_l=55):
    """
    改进后的预处理函数：
    1. 添加去重逻辑
    2. 验证数据完整性
    """
    # 检查输入
    if 'text' not in tokenized_data.columns or 'dom' not in tokenized_data.columns:
        raise ValueError("输入数据需要包含'text'和'dom'列")

    # 分词和编码
    # tokenized_data = get_tokenized(data)
    features = []
    labels = []

    # 确保每个样本独立处理
    for _, row in tokenized_data.iterrows():
        words = row['text']
        indices = vocab.to_indices(words)
        if len(indices) < max_l:
            padding = [vocab['<pad>']] * (max_l - len(indices))  # 仅用pad
            indices += padding
        else:
            indices = indices[:max_l]
        features.append(indices)
        labels.append(row['dom'])

    # 转换为tensor并去重
    features = torch.tensor(features)
    labels = torch.tensor(labels)

    # 重要！去除完全相同的样本
    unique_features, unique_indices = torch.unique(features, dim=0, return_inverse=True)
    unique_labels = labels[torch.unique(unique_indices)]

    return unique_features, unique_labels

class TextDataset(Dataset):
    """自定义数据集类"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def data_iterator(features, labels, batch_size=32, shuffle=True):
    """
    创建数据迭代器

    参数:
        features: 特征张量 (torch.Tensor)
        labels: 标签张量 (torch.Tensor)
        batch_size: 批量大小 (默认32)
        shuffle: 是否随机打乱数据 (默认True)

    返回:
        DataLoader迭代器
    """
    # 确保特征和标签长度一致
    assert len(features) == len(labels), "特征和标签数量不匹配"

    # 创建数据集
    dataset = TextDataset(features, labels)

    # 创建数据加载器
    data_iter = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False  # 不丢弃最后不足batch_size的批次
    )

    return data_iter

def shuffl_split_data(df, test_size = 0.2):
    """加载并预处理数据"""

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,       # 测试集比例
        stratify=df['dom']
        # random_state=42      # 随机种子
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# 示例用法
if __name__ == "__main__":
    # 模拟数据 (100个样本，特征长度70)
    features = torch.randint(0, 1000, (100, 70))
    labels = torch.randint(0, 5, (100,))

    # 创建迭代器
    data_iter = data_iterator(features, labels, batch_size=16)

    # 测试迭代
    for batch_idx, (batch_x, batch_y) in enumerate(data_iter):
        print(f"批次 {batch_idx}: 特征形状 {batch_x.shape}, 标签形状 {batch_y.shape}")
        # 这里可以添加训练代码
        break  # 只演示第一个批次