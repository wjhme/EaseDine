from .utils import del_keywords, food_feature, cls, similarity, nlp_recommend
import pandas as pd 
from tqdm import tqdm
tqdm.pandas()
import faiss
import numpy as np
import ast
import torch

import os
current_dir = os.getcwd()
os.chdir(current_dir)

def recommend(data_path, save_path):

    df = pd.read_csv(data_path, sep='\t')
    # 筛选出点餐指令
    df_query = df[df['dom']==1]

    #删除keywords
    df_query['text'] = df_query['text'].progress_apply(del_keywords)

    # 大模型提取菜品特征
    df_query['food_feature'] = df_query['text'].progress_apply(food_feature)

    # 相似度分类
    # 菜品库-读取时将list转回tensor
    df_food = pd.read_csv("QUE/商品-清洗.txt")
    df_food['cls'] = df_food['cls'].apply(lambda x: torch.tensor(ast.literal_eval(x))) 

    #转换词向量
    df_query['cls'] = df_query['food_feature'].progress_apply(cls)

    # 菜品库向量FAISS 存储
    # 把向量堆叠成 numpy 数组（shape: N x dim）
    embeddings = np.stack(df_food['cls'].values).astype('float32')
    dim = embeddings.shape[1]

    # 创建索引（使用 L2 距离）
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    def query(food_cls):
        query_vector = food_cls.reshape(1, -1)
        D, I = index.search(query_vector, k=1)  # D: 距离, I: 索引
        formatted = df_food.loc[I[0][0],'item_name']
        return formatted

    # 最终推荐菜品与提取特征间的相似度
    #查询
    df_query['Query_faiss'] = df_query.progress_apply(lambda row: query(row['cls']),axis=1)
    df_query['cls1'] = df_query['Query_faiss'].progress_apply(cls)
    df_query['similarity'] = df_query.progress_apply(lambda row: similarity(row['cls'], row['cls1']), axis=1)

    df_query_1 = df_query[df_query['similarity']>=0.98]
    df_query_2 = df_query[df_query['similarity']<0.98]

    #第一部分
    df_query_1['recommend'] = df_query_1['Query_faiss']

    #第二部分
    def query2(food_cls,feature):
        query_vector = food_cls.reshape(1, -1)
        D, I = index.search(query_vector, k=5)  # D: 距离, I: 索引
        formatted = list(df_food.loc[s,'item_name'] for i, s in enumerate(I[0]))
        #特殊描述添加召回通道
        if any(keyword in feature for keyword in ['别太油','不油腻','不要油','不辣不咸','少放点味精','少放油盐','少油','少盐','少放调料','少放点调料','别整油炸']):
            formatted += ['清炒时蔬菜','不油腻的油麦菜','清炒空心菜']
        if any(keyword in feature for keyword in ['软和的','煮的软','软和点','软和些','软软的','软烂的','不费牙','软糯的','软的','不硌牙','烂糊点','易嚼','炖得烂','软乎','好嚼','温乎的饭菜','炖得久的菜','易嚼的','好咬的','牙口不好的','不要硬邦邦的','煮烂点']):
            formatted += ['软香大米饭','一家亲手擀面','肉丝烂糊白菜','小份烂糊白菜','土豆豆角焖面','冬瓜粉条炖肉']
        if any(keyword in feature for keyword in ['暖胃','养胃','好消化','老年人','易消化']):
            formatted += ['养胃小米粥','白菜豆腐养生汤','小米暖胃粥']
        if any(keyword in feature for keyword in ['汤汤水水','汤水多','带汤的','带汤水']):
            formatted += ['西红柿紫菜汤','白菜豆腐养生汤','羊肉汤清汤']
        if any(keyword in feature for keyword in ['不要辣','别放辣','不要放辣椒','少放辣','不辣','不放辣椒']):
            formatted += ['家常炒鸡不辣','清炒时蔬菜','鲫鱼豆腐汤','清淡菠菜粥']
        if any(keyword in feature for keyword in ['家常便饭','简单点','家常菜','家常的','家常味道']):
            formatted += ['家常手擀面','一荤两素加米饭','小份家常炖菜']
        if any(keyword in feature for keyword in ['易下咽','顺口','好咽']):
            formatted += ['爽囗小米粥','爽口什锦小菜','清爽蛋炒饭']    
        if any(keyword in feature for keyword in ['有营养','健康','养生的','补身子','健康点']):
            formatted += ['大份营养鸡汤','白菜豆腐养生汤','健康养胃小米粥']
        if any(keyword in feature for keyword in ['热乎的','温乎的','暖和']):
            formatted += ['温热南瓜汤','热桂圆红枣汤','传统热汤面','小份烂糊白菜']    
        if any(keyword in feature for keyword in ['主食','软乎面食','软软的面食']):
            formatted += ['软香大米饭','一家亲手擀面']    
        return formatted
    
    #查询
    df_query_2['Query_faiss'] = df_query_2.progress_apply(lambda row: query2(row['cls'],row['food_feature']),axis=1)
    
    #大模型推荐
    df_query_2['recommend'] = df_query_2.progress_apply(lambda row: nlp_recommend(row['text'], row['Query_faiss']), axis=1)
    
    # faiss审核确保输出
    def query_post(recommend):
    #返回最相近的菜品
        food_cls = cls(recommend)
        query_vector = food_cls.reshape(1, -1)
        D, I = index.search(query_vector, k=1)  # D: 距离, I: 索引
        return df_food.loc[I[0][0],'item_name']
    #查询
    df_query_2['recommend'] = df_query_2['recommend'].progress_apply(query_post)
    #拼接
    df_query_0 = pd.concat([df_query_1, df_query_2], axis=0)

    # 提交文件格式
    df_result = pd.merge(df, df_query_0[['uuid','recommend']], how='left', on='uuid')
    df_result.columns

    df_result.to_csv(save_path, sep='\t', header=None, index=False, na_rep='')