import os
import time
# import dashscope
from openai import OpenAI
# from QUE.utils import food_feature
import pandas as pd


sentence = "来份东北香米饭吧"
# messages = [
#         {'role': 'system',
#          'content': '''你是一个点餐指令文本分析助手。输入为用户语音指令,输出为从用户语音指令中提取的菜名,如果没有具体菜名,则从句子中提取有关菜品的特征描述。
#          例如：
#         user:天猫精灵来碗红枣桂圆养胃粥要热乎的  assistant：红枣桂圆养胃粥  
#         user:天猫精灵少油少盐的就行  assistant:少油少盐'''},
#         {'role': 'user', 'content':sentence}]

# t0 = time.time()
# response = dashscope.Generation.call(
#     # 若没有配置环境变量,请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key=os.getenv('DASHSCOPE_API_KEY'),
#     model="deepseek-v3",  # 此处以 deepseek-r1 为例,可按需更换模型名称。
#     messages=messages,
#     # result_format参数不可以设置为"text"。
#     result_format='message'
# )

# print("=" * 20 + "最终答案" + "=" * 20)
# print(response.output.choices[0].message.content)

# # print(response.output.choices[0].message.content)
# # print(food_feature(sentence))
# print(f"{time.time() - t0:.4f}")

# # ============================= 批量处理 ====================================
# df = pd.read_csv("/mnt/disk/wjh23/EaseDine/ASR/FunASR/A_audio_results/FunASR_A_audio_best_model_189_train_val_avg.txt", sep="\t",header=None,names=['uuid','text'])[3000:4000]
# sentences = df['text'].tolist()
# llm_sentence = []
# num = 0
# for sentence in sentences:
#     num+=1
#     if num % 100==0:
#         print(f"{num}/{len(sentences)}")
#     message = [
#     {'role': 'system',
#     'content': '''你是一个语音识别文本纠错助手,擅长对天猫语音助手识别的文本进行纠错。'''},
#     {'role': 'user', 'content':f'''
#     请对输入的语音识别文本进行同音字纠错,严格按照以下规则处理：
#     1.纠正同音字错误的内容,对纠正前后变化的内容检查拼音是否一致进行验证：
#     **同音字判定标准**：
#     - 仅修正普通话中**发音完全相同**（声母/韵母/声调全一致）的词汇
#     - 排除儿化音、轻声等特殊发音差异（如"这儿"≠"这"）
#     - 示例错误："养位→养胃"（wèi→wèi）；正确保留："拉屎→蜡石"（shǐ≠shí）
#     2.保持原文语序和表达方式不变,原文的重复表达无需纠正,输出前检查文本长度纠正前后需保持不变；
#     3.不确定的词语保持原样；
#     4.方言、口语化表达不纠正（如：吾/我、侬/你）；
#     5.部分文本会存在“屎、尿、屁等不文明用词”,这类内容不纠正,保持不变；
#     6.部分文本会出现中文歌曲名,如出现同音字错误进行纠正,否则保持不变,并检查对应的歌手名是否正确；
#     7.部分文本会出现语气词,保持不变,如：吧、嗯、额、呃等
#     8.比较明确的常规词组错误,可进行修改,如：排击炮->迫击炮、
#     9.输出只需包含纠正后的文本,无需解释。
#     示例：
#     输入：天猫精灵来碗红枣桂圆养位粥要热乎的
#     输出：天猫精灵来碗红枣桂圆养胃粥要热乎的

#     输入：来份香辣手斯包菜锅吧
#     输出：来份香辣手撕包菜锅吧
 
#     输入：天猫精灵四六十四减十等于几
#     输出：天猫精灵四六十四减十等于几
 
#     输入：天猫精灵给我放个拉屎的声音
#     输出：天猫精灵给我放个拉屎的声音

#     输入：{sentence}
#     '''}]
#     client = OpenAI(
#     # 若没有配置环境变量,请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key="sk-01730ba2d0ac444ab5d2271feef413f6", 
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     )
#     completion = client.chat.completions.create(
#     model="deepseek-v3", # 此处以qwen-plus为例,可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     messages=message,
#     )
#     ans = completion.choices[0].message.content
#     llm_sentence.append(ans)    
  
# df['llm_text'] = llm_sentence

# # 处理纠错后的文本：删除标点符号、数字；淘汰文本长度变化的文本
# df["llm_text"] = df["llm_text"].str.replace(r'[\s,。？！,.?!\d+]', '', regex=True)
# # 当text和llm_text长度不同时,用text替换llm_text
# for idx, row in df.iterrows():
#     text_len = len(str(row['text']))
#     llm_text_len = len(str(row['llm_text']))

#     if text_len != llm_text_len:
#         df.at[idx, 'llm_text'] = row['text']

# # 保存纠正前后数据
# df.to_csv("/mnt/disk/wjh23/EaseDine/asr_llm_results/A_audio_best_model_189_train_val_avg_deepseek_r1_1000_2000.txt",sep="\t",index=False)

# # # 保存纠正前后不同的数据
# # diff = df[df['llm_text']!=df['text']]
# # diff.to_csv("/mnt/disk/wjh23/EaseDine/asr_llm_results/diff_deepseek_r1.txt",sep="\t",index=False)

# # # 只保存纠正后数据
# # df.drop('text',axis=1,inplace=True)
# # df.to_csv("/mnt/disk/wjh23/EaseDine/asr_llm_results/A_audio_best_model_189_train_val_avg_deepseek_r1_0_1000.txt",sep="\t",index=False,header=None)

# ============================= 单独测试 ====================================
sentence = "我要唱是二志呀的一封时光"
message = [
{'role': 'system',
'content': 
'''
# 文本纠错规则手册

您是天猫语音文本纠错专家，请严格按优先级执行以下规则：

一、强制保留规则（最高优先级）
1. 遇到以下情况直接返回原文：
   - 数学运算相关文本：加减乘除
   - 含方言词汇：吾/侬/好伐/今朝
   - 包含排泄词汇：屎/尿/屁/粑粑
   - 含语气词：吧/嗯/额/呃（禁止增删）

二、专项纠错规则
2. 点歌指令（触发词：来首/放一首/想听/唱/放一曲...）
   - 仅修正发音相同（拼音+声调一致）的歌曲/歌手名
   - 示例： 
     × "橘花台" → √ "菊花台"（jú同音）
     × "青花磁" → √ "青花瓷"（cí同音）
     × 不可改"双截棍"为"双节棍"（非发音错误）

3. 点餐指令（触发词：来份/来碗/要一碗...）
   - 仅修正发音相同的菜品名
   - 示例：
     × "宫爆鸡丁" → √ "宫保鸡丁"
     × 不可改"鱼香肉丝"为"余香肉丝"

4. 专有名词修正（前序规则不适用时）
   - 仅修正发音相同的明显错误
   - 示例：
     × "排击炮" → √ "迫击炮"
     × 不可改"微信"为"围信"

三、操作限制
5. 严格禁止：
   - 修改非发音错误（如近义词/形近字）
   - 拆分/合并词语（如"的士"改"出租车"）
   - 添加/删除任何字符（包括标点）

6. 最终校验：
   - 修改后文本长度必须与原文完全一致
   - 若长度变化/不确定错误，返回原文

请直接返回纠正后的文本或原文.
'''},
{'role': 'user', 'content':sentence}]
client = OpenAI(
# 若没有配置环境变量,请用百炼API Key将下行替换为：api_key="sk-xxx",
api_key="sk-01730ba2d0ac444ab5d2271feef413f6", 
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
model="deepseek-r1", # qwen3-235b-a22b | deepseek-v3此处以qwen-plus为例,可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
messages=message,
)
ans = completion.choices[0].message.content
print(ans)


f'''
    请对输入的语音识别文本进行同音字纠错,严格按照以下规则处理：
    1.纠正同音字错误的内容,对纠正前后变化的内容检查拼音是否一致进行验证：
    **同音字判定标准**：
    - 仅修正普通话中**发音完全相同**（声母/韵母/声调全一致）的词汇
    - 排除儿化音、轻声等特殊发音差异（如"这儿"≠"这"）
    - 示例错误："养位→养胃"（wèi→wèi）；正确保留："拉屎→蜡石"（shǐ≠shí）
    2.保持原文语序和表达方式不变,原文的重复表达无需纠正,输出前检查文本长度纠正前后需保持不变；
    3.不确定的词语保持原样；
    4.方言、口语化表达不纠正（如：吾/我、侬/你）；
    5.部分文本会存在“屎、尿、屁等不文明用词”,这类内容不纠正,保持不变；
    6.部分文本会出现中文歌曲名,如出现同音字错误进行纠正,否则保持不变,并检查对应的歌手名是否正确；
    7.部分文本会出现语气词,保持不变,如：吧、嗯、额、呃等
    8.比较明确的常规词组错误,可进行修改,如：排击炮->迫击炮、
    9.输出只需包含纠正后的文本,无需解释。
    示例：
    输入：天猫精灵来碗红枣桂圆养位粥要热乎的
    输出：天猫精灵来碗红枣桂圆养胃粥要热乎的

    输入：来份香辣手斯包菜锅吧
    输出：来份香辣手撕包菜锅吧
 
    输入：天猫精灵四六十四减十等于几
    输出：天猫精灵四六十四减十等于几
 
    输入：天猫精灵给我放个拉屎的声音
    输出：天猫精灵给我放个拉屎的声音

    语音识别文本：{sentence}

你是一个文本纠错助手,擅长对天猫语音助手识别文本中存在的错误内容进行纠正.请按照下面的规则对输入的文本纠正,只返回纠正后的文本或原文,不要返回其他无关内容或解释：
1、点歌相关文本,关键词有【来首,放一首,想听,唱,放一曲...】:
检查文本中提到的歌曲名和歌手名是否有误,如果发音相同的字出现错误进行纠正,禁止更换歌手名或歌曲名,不确定的不纠正;
例如：我要听周杰伦的橘花台,其中"橘花台"存在同音字错误,正确为"菊花台",纠正为:我要听周杰伦的菊花台
2、数学计算文本,关键词【除以,乘,等于,加,减】:
该类型文本不纠正,直接返回原文;
3、点餐文本或者包含菜品的文本,关键词【来份,来碗,来一份,要一碗】:
检查文本中提到的菜品名称是否有误,如果发音相同的字出现错误进行纠正,不要更换菜品名称,不确定的不纠正;
4、方言或口语化的文本,关键词【吾,侬,好伐,今朝】:
该类型文本中的方言表述不要修改;
5、其他文本:
检查文本中的常用专有名词是否存在明显错误,如果发音相同的字出现错误进行纠正,不要更换其他专有名词,不确定的不纠正;
例如：排击炮(专有名词“迫击炮”)
6、部分文本会存在“屎、尿、屁、粑粑等不文明用词”,这类内容不纠正,保持不变;
7、部分文本会出现语气词,如：吧、嗯、额、呃等,不要对这些语气词进行增删;
8、最后检查修改后的文本长度是否发生变化,如果变化,返回对应的原文.
'''
