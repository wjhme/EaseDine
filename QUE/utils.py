from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import dashscope
import time
import torch

def del_keywords(s):
    for sub in ["天猫精灵", "来份",'要份','要一份','来点','来碗','吧','来一份','要个','要碗']:
        s = s.replace(sub, "")
    return s

def food_feature(sentence):
    message = [
        {'role': 'system',
        'content': '''你是一个文本分析助手，任务是从用户的输入中提取菜品描述信息，尽量精简。输出必须仅包含用户原始文本中出现的词语，不得添加或扩展内容。
        示例1：
        user: 红枣桂圆养胃粥要热乎的  
        assistant: 热乎的红枣桂圆养胃粥
        示例2：
        user: 少油少盐的就行  
        assistant: 少油少盐
        示例3：
        user: 热乎养胃小米粥  
        assistant: 热乎养胃小米粥
        示例4：
        user: 家常手撕包菜  
        assistant: 家常手撕包菜
        示例5：
        user: 阿拉三色杂粮喷香米饭快点好伐
        assistant: 三色杂粮喷香米饭
        '''},
        {'role': 'user', 'content':sentence}]
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-606b64b8196d4f119a7fb1789c86ac7c", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen2.5-32b-instruct", 
        messages=message,
        )
    return completion.choices[0].message.content

def nlp_recommend(food_name,food_choice):
    time.sleep(1.2)
    input = food_name+'+'+str(food_choice)
    message = [
        {'role': 'system', 'content': '''你是一个智能点餐助手。给你一个点餐文本，以及可能的菜品列表。你的任务是从菜品列表中选出最符合点餐文本的菜品，并且只返回该菜品名称。
                    例如：
                    user：天猫精灵要个家常花菜炒肉片吧+[花菜炒肉片, 菜花炒肉片, 家常炒花菜, 花菜家常炒, 花菜炒肉丝]
                    assistant:花菜炒肉片'''},
        {'role': 'user', 'content':input}]
    response = dashscope.Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-606b64b8196d4f119a7fb1789c86ac7c",
        model="qwen2.5-32b-instruct-ft-202505160052-9abd", 
        messages=message,
        result_format='message'
        )
    return response["output"]["choices"][0]["message"]["content"]


# 获取BAAI绝对路径
path = str(Path(__file__).parent / "BAAI") 

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path)
#转换词向量函数
def cls(text): 
    input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**input).last_hidden_state
    cls_embedding = output[:, 0, :]
    cls_embedding = cls_embedding[0]
    return cls_embedding

def similarity(cls1,cls2):
    similarity = torch.cosine_similarity(cls1, cls2, dim=0)
    return similarity.item()
