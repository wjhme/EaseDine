from utils import food_class, food_feature, nlp_recommend, cls, query
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
import faiss
import pandas as pd
import numpy as np
import torch
import os
import time

# 读取文件
df_query = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/点餐指令_测试.txt", sep='\t')

# df_query = df[df['dom']==1]
#df_query = df_query.head(10)


# 大模型提取文本点餐信息
df_query['food_feature'] = df_query['raw_text'].progress_apply(food_feature)
df_query[['text', 'class']] = df_query['food_feature'].str.split('-', expand=True)
df_query['class'] = df_query['class'].astype(int) 
df_query.drop('food_feature', axis=1, inplace=True)  # 删除原列
df_query.to_csv("点餐指令_测试_提取特征_temp.txt",sep="\t",index=False)

# # 点餐指令清晰数据
# df_query_1 = df_query[df_query['class']==1]
# # 点餐指令模糊数据
# df_query_0 = df_query[df_query['class']==0]

# 召回
# 读取时将list转回tensor
import ast
t0 = time.time()
df_food = pd.read_csv("/mnt/disk/wjh23/EaseDine/QUE/food_embeding.txt")
# df_food['cls'] = df_food['cls'].apply(lambda x: torch.tensor(ast.literal_eval(x)))
# 将字符串解析为 NumPy 数组
numpy_data = np.array([np.fromstring(x.strip("[]"), sep=',') for x in df_food['cls']])
# 转 PyTorch 张量
df_food['cls'] = [torch.from_numpy(arr) for arr in numpy_data]
print(f"转 PyTorch 张量用时：{time.time() - t0:.4f}s")

# df_query_1['cls'] = df_query_1['food'].progress_apply(cls)
df_query = cls(df_query)
# 将字符串解析为 NumPy 数组
numpy_data = np.array([np.fromstring(x.strip("[]"), sep=',') for x in df_query['cls']])
# 转 PyTorch 张量
df_query['cls'] = [torch.from_numpy(arr) for arr in numpy_data]

# faiss 存储
# 把向量堆叠成 numpy 数组（shape: N x dim）
embeddings = np.stack(df_food['cls'].values).astype('float32')
dim = embeddings.shape[1]

# 创建索引（使用 L2 距离）
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print(f"向量数量：{index.ntotal}")

#查询
df_query['Query_faiss'] = df_query.progress_apply(lambda row: query(index, row['cls'], df_food),axis=1)

## nlp推荐
df_query['nlp_recommend'] = df_query.progress_apply(lambda row: nlp_recommend(row['text'], row['Query_faiss']), axis=1)
df_query.drop(['tokenized', 'cls'], axis=1, inplace=True) 
df_query.to_csv("df_query_results.txt",sep="\t",index=False)

# def faiss_1(s):
#     return s[0]  # 输出: 花菜炒肉片

# df_query['faiss_1'] = df_query['Query_faiss'].progress_apply(faiss_1)

# # faiss审核确保输出
# df_query['nlp_cls'] = df_query['nlp_recommend'].progress_apply(cls)

# def query_post(food_cls):
# #返回最相近的菜品
#     query_vector = food_cls.reshape(1, -1)
#     D, I = index.search(query_vector, k=1)  # D: 距离, I: 索引
#     return df_food.loc[I[0][0],'item_name']

# #查询
# df_query['nlp_post'] = df_query['nlp_cls'].progress_apply(query_post)

# df_query['nlp_post_cls'] = df_query['nlp_post'].progress_apply(cls)

# # 最终推荐菜品与提取特征间的相似度
# def similarity(cls1,cls2):
#     similarity = torch.cosine_similarity(cls1, cls2, dim=0)
#     return similarity.item()

# df_query['similarity'] = df_query.progress_apply(lambda row: similarity(row['cls'], row['nlp_post_cls']), axis=1)

# df_query[df_query['feature_class']=='1']['similarity'].mean()

# # 保存结果
# df_result = pd.merge(df, df_query[['uuid','nlp_post']], how='left', on='uuid')

# df_result.to_csv(r".\result\Results506.txt", 
#           sep='\t', 
#           header=None, 
#           index=False, 
#           na_rep='')