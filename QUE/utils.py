
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

import sys
from pathlib import Path

# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录（假设项目根目录是 EaseDine 的父目录）
project_root = current_file.parent.parent  # 根据实际层级调整
# 将项目根目录添加到 Python 路径
sys.path.append(str(project_root))
from DOM.utils import process_data, Embeding
import pandas as pd
import numpy as np
import json


#后续尝试批量请求加速时间
def food_feature(sentence):
    message = [
        {'role': 'system',
         'content': '''你是一个点餐指令文本分析助手。'''},
        {'role': 'user', 'content':f'''解读输入文本并按照以下要求回复：
        输入为用户语音指令；
        输出为从用户语音指令中提到的【菜品名】及标签【1】。如果没有具体菜名，则输出指令中提到的有关菜品的【特征描述】及标签【0】(只提取菜品特征的描述，如：好消化的、少盐的等)，以原文本描述为主，不要添加其他描述；
        注意：只返回【菜品名】或菜品的【特征描述】中的一个结果，无需输出相关解释及其他内容。
        例如：
        输入:天猫精灵来碗红枣桂圆养胃粥要热乎的  输出:红枣桂圆养胃粥-1  
        输入:天猫精灵少油少盐的就行  输出:少油少盐-0
        输入:天猫精灵来份香辣手撕巴菜吧  输出:香辣手撕包菜-1
        输入：{sentence}
         '''}]
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-01730ba2d0ac444ab5d2271feef413f6", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen2.5-32b-instruct", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=message,
        )
    return completion.choices[0].message.content

def food_class(sentence):
    message = [
        {'role': 'system', 
         'content': '''你要做一个二分类任务，如果输入为菜品，返回1，如果输入为对菜品的模糊描述，返回0。
        例如:
        user： 红枣桂圆养胃粥 assistant：1  
        user：软和的 assistant：0'''},
        {'role': 'user', 'content':sentence}]
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-01730ba2d0ac444ab5d2271feef413f6", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen2.5-32b-instruct", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=message,
        )
    return completion.choices[0].message.content

def nlp_recommend(food_name,food_choice):
    input = food_name+'+'+str(food_choice)
    message = [
        {'role': 'system', 'content': '''你是一个智能菜品推荐助手。给你一个点餐文本，以及候选菜品列表。你的任务是从菜品列表中选出与点餐文本最匹配的菜品。
        注意：只返回推荐菜品名称，解释内容及其他无关内容不要输出。
        例如：
        user:天猫精灵要个家常花菜炒肉片吧+[花菜炒肉片, 菜花炒肉片, 家常炒花菜, 花菜家常炒, 花菜炒肉丝]
        assistant:花菜炒肉片'''},
        {'role': 'user', 'content':input}]
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-606b64b8196d4f119a7fb1789c86ac7c", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen2.5-32b-instruct", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=message,
        )
    return completion.choices[0].message.content

#转换词向量函数
# 初始化模型和分词器

path = "BAAI/bge-large-en"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path)
model.eval()  # 设置为评估模式
# def cls(text): 
#     input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         output = model(**input).last_hidden_state
#     cls_embedding = output[:, 0, :]
#     cls_embedding = cls_embedding[0]
#     return cls_embedding

def cls(data): 
    em = Embeding()
    DIM = 300
    data = process_data(data, stop_words="/mnt/disk/wjh23/EaseDine/DOM/stopwords/que_stopwords.txt",drop_duplicates=False)

    data_embeding = em.get_word2vec(data,f"/mnt/disk/wjh23/EaseDine/DOM/embeding_models/que_word2vec/word2vec_model_1_{DIM}.bin")

    data['cls'] = [json.dumps(vec.tolist()) for vec in data_embeding]
    return data

def query(index, food_cls, df_food):
    query_vector = food_cls.reshape(1, -1).numpy()
    query_vector = np.stack(query_vector).astype('float32')
    D, I = index.search(query_vector, k=5)  # D: 距离, I: 索引
    formatted = list(df_food.loc[s,'item_name'] for i, s in enumerate(I[0]))
    print("formatted:",formatted)
    return formatted