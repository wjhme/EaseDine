import torch
from TextCNN import TextCNN
import pandas as pd
import torch.nn as nn
from EaseDine.DOM.utils import get_tokenized, get_vocab, process_data, data_iterator, shuffl_split_data

df = pd.read_csv("../train.txt",sep="\t")
# 分词处理
df = get_tokenized(df)
# 去重
df = df.drop_duplicates(subset=['text'], keep='first')
df.to_csv("processed_train.txt",sep="\t",index=False)
# duplicates = df[df.duplicated(subset=['text'])]
# print(f"完全重复的行数: {len(duplicates)}")
# print(duplicates)
train_df, test_df = shuffl_split_data(df, test_size = 0.2)

print("类别分布 train_df:\n", train_df['dom'].value_counts())
print("类别分布 test_df:\n", test_df['dom'].value_counts())

# 自定义训练词向量
from gensim.models import Word2Vec
sentences = train_df['text'].tolist()
model = Word2Vec(sentences, vector_size=200, window=3, min_count=1)
model.save("custom_word2vec.model")

# 创建词典
vocab = get_vocab(train_df)

# 预处理数据
train_features, train_labels = process_data(train_df,vocab, max_l = 65)
test_features, test_labels = process_data(test_df,vocab, max_l = 65)
print("train_features:",len(train_features))
print("test_features:",len(test_features))

# 创建数据迭代器
batch_size = 64
train_iter = data_iterator(train_features, train_labels, batch_size=batch_size)
test_iter = data_iterator(test_features, test_labels, batch_size=batch_size)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)
embed_dim, kernel_sizes, num_filters = 200, [2, 3, 4, 5], 128

# 加载自定义词向量模型
w2v_model = Word2Vec.load("custom_word2vec.model")
# 步骤2：构建权重矩阵
weight_matrix = torch.zeros((len(vocab), embed_dim))
for word, idx in vocab.token_to_idx.items():
    if word in w2v_model.wv:
        weight_matrix[idx] = torch.FloatTensor(w2v_model.wv[word])
    else:
        weight_matrix[idx] = torch.randn(embed_dim)  # 随机初始化未登录词

model = TextCNN(vocab_size = len(vocab),
                embed_dim = embed_dim,
                num_classes = 2,
                kernel_sizes = kernel_sizes,
                num_filters = num_filters,
                dropout=0.5,
                weight_matrix = weight_matrix,
                pretrained_embeddings=True
                ).to(device)


# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
# 动态学习率 (验证损失不下降时衰减)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加L2正则
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # 余弦退火

# # 训练循环
# # 添加早停机制
# best_acc = 0
# patience = 5
# for epoch in range(20):
#     model.train()
#     for batch_x, batch_y in train_iter:
#         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#         optimizer.zero_grad()
#         outputs = model(batch_x)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         # 梯度裁剪
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#         # 打印梯度信息
#         total_norm = 0
#         for p in model.parameters():
#             if p.grad is not None:
#                 param_norm = p.grad.data.norm(2)
#                 total_norm += param_norm.item() ** 2
#         total_norm = total_norm ** (1. / 2)
#         # print(f"梯度范数: {total_norm:.4f}")
#         optimizer.step()
#
#     # 验证
#     model.eval()
#     with torch.no_grad():
#         correct, total = 0, 0
#         for batch_x, batch_y in test_iter:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             outputs = model(batch_x)
#             _, predicted = torch.max(outputs, 1)
#             total += batch_y.size(0)
#             correct += (predicted == batch_y).sum().item()
#         print(f"Epoch {epoch}, Test Acc: {correct / total:.4f}")
#     current_acc = correct / total
#     # 验证后添加
#     if current_acc > best_acc:
#         best_acc = current_acc
#         patience_counter = 0
#     else:
#         patience_counter += 1
#         if patience_counter >= patience:
#             print("Early stopping!")
#             break
#     scheduler.step()  # 更新学习率
#
# # 检查训练/测试集是否有重叠
# train_set = set(tuple(x) for x in train_features.numpy())
# test_set = set(tuple(x) for x in test_features.numpy())
# print(f"训练集和测试集重叠样本数: {len(train_set & test_set)}")