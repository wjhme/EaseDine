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

def llm_corrector(data_path, save_corrected_path, save_diff_path, batch_size=4):
    df = pd.read_csv(data_path, sep="\t", header=None, names=['uuid', 'text'])
    # df = pd.read_csv(data_path, sep="\t")
    sentences = df['text'].tolist()
    
    # 将句子分成4个批次
    batches = np.array_split(sentences, batch_size)
    total_batches = len(batches)
    
    llm_sentence = [None] * len(sentences)
    
    # 使用线程池并行处理批次
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for i, batch in enumerate(batches):
            futures.append(executor.submit(process_batch, batch.tolist(), i + 1, total_batches))
        
        for future in concurrent.futures.as_completed(futures):
            batch_num, results = future.result()
            start_idx = (batch_num - 1) * len(results)
            end_idx = start_idx + len(results)
            llm_sentence[start_idx:end_idx] = results
    
    df['llm_text'] = llm_sentence

    # 处理纠错后的文本:删除标点符号、数字;淘汰文本长度变化的文本
    df["llm_text"] = df["llm_text"].str.replace(r'[\s，。:："？！,.?!\d+]', '', regex=True)
    
    # 当text和llm_text长度不同时，用text替换llm_text
    for idx, row in df.iterrows():
        text_len = len(str(row['text']))
        llm_text_len = len(str(row['llm_text']))

        if text_len != llm_text_len:
            df.at[idx, 'llm_text'] = row['text']

    # 保存纠正前后不同的数据
    diff = df[df['llm_text']!=df['text']]
    diff.to_csv(save_diff_path, sep="\t", index=False)

    # 只保存纠正后数据
    df.drop('text', axis=1, inplace=True)
    df.to_csv(save_corrected_path, sep="\t", index=False, header=None)