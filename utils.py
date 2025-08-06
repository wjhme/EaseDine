import pandas as pd
from openai import OpenAI
import concurrent.futures
import numpy as np

def process_batch(batch_sentences, batch_num, total_batches):
    llm_sentence = []
    client = OpenAI(
        api_key="sk-01730ba2d0ac444ab5d2271feef413f6", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    for i, sentence in enumerate(batch_sentences):
        if (i + 1) % 10 == 0 or i == len(batch_sentences) - 1:
            print(f"Batch {batch_num}/{total_batches}: Processing {i + 1}/{len(batch_sentences)}")
        
        message = [
            {'role': 'system',
             'content': '''你是一个语音识别文本纠错助手，擅长对天猫语音助手识别的文本进行纠错。'''},
            {'role': 'user', 'content':f'''
            请对输入的语音识别文本进行同音字纠错，严格按照以下规则处理:
            1.纠正同音字错误的内容，对纠正前后变化的内容检查拼音是否一致进行验证:
            **同音字判定标准**:
            - 仅修正普通话中**发音完全相同**(声母/韵母/声调全一致)的词汇
            - 排除儿化音、轻声等特殊发音差异(如"这儿"≠"这")
            - 示例错误:"养位→养胃"(wèi→wèi);正确保留:"拉屎→蜡石"(shǐ≠shí)
            2.保持原文语序和表达方式不变，原文的重复表达无需纠正，输出前检查文本长度纠正前后需保持不变;
            3.不确定的词语保持原样;
            4.方言、口语化表达不纠正(如:吾/我、侬/你);
            5.部分文本会存在"屎、尿、屁等不文明用词"，这类内容不纠正，保持不变;
            6.部分文本会出现中文歌曲名，如出现同音字错误进行纠正，否则保持不变，并检查对应的歌手名是否正确;
            7.部分文本会出现语气词，保持不变，如:吧、嗯、额、呃等
            8.比较明确的常规词组错误，可进行修改，如:排击炮->迫击炮、
            9.输出前检查下是否符合上述的要求，输出只需包含纠正后的文本，无需解释及其他无关内容。
            示例:
            输入:天猫精灵来碗红枣桂圆养位粥要热乎的
            输出:天猫精灵来碗红枣桂圆养胃粥要热乎的

            输入:来份香辣手斯包菜锅吧
            输出:来份香辣手撕包菜锅吧
        
            输入:天猫精灵四六十四减十等于几
            输出:天猫精灵四六十四减十等于几
        
            输入:天猫精灵给我放个拉屎的声音
            输出:天猫精灵给我放个拉屎的声音

            输入:{sentence}
            '''}
        ]
        
        try:
            completion = client.chat.completions.create(
                model="deepseek-v3",# qwen-plus  deepseek-v3
                messages=message,
            )
            ans = completion.choices[0].message.content
            llm_sentence.append(ans)
        except Exception as e:
            print(f"Error processing sentence: {sentence}. Error: {str(e)}")
            llm_sentence.append(sentence)  # 如果出错，保留原句
    
    return batch_num, llm_sentence
