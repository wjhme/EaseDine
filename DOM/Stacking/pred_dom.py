import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from Classifiers import DOM

# # 获取当前文件的绝对路径
# current_file = Path(__file__).resolve()
# # 获取项目根目录（假设项目根目录是 EaseDine 的父目录）
# project_root = current_file.parent.parent.parent  # 根据实际层级调整
# DOM_path = current_file.parent.parent
# # 将项目根目录添加到 Python 路径
# sys.path.append(str(project_root))
# from DOM.utils import process_data, Embeding

# def pre_dom(recognized_path, save_path):
#     '''根据识别的文本预测DOM类别'''

#     # 读取数据
#     recognized_data = pd.read_csv(recognized_path,sep="\t",header=None,names=['uuid','text'])
#     # 数据预处理
#     processed_data = process_data(recognized_data, drop_duplicates = False)

#     # 生成词向量
#     em = Embeding(processed_data)
#     # Word3Vec
#     DIM = 100
#     word2vec_embeding = em.get_word2vec(f"{str(DOM_path)}/embeding_models/word2vec/word2vec_model_0_{DIM}.bin")

#     # LDA 主题特征
#     num_topics = 20
#     lda_model_path = f"{str(DOM_path)}/embeding_models/lda/lda_model_{num_topics}.model"
#     lda_dictionary_path = f"{str(DOM_path)}/embeding_models/lda/lda_dictionary_{num_topics}.gensim"
#     lda_corpus_path = f"{str(DOM_path)}/embeding_models/lda/lda_corpus_{num_topics}.mm"
#     lda_embeding = em.get_lda(num_topics,lda_model_path,lda_dictionary_path,lda_corpus_path)

#     # 按行拼接（沿 axis=1 水平拼接）
#     features = np.hstack([lda_embeding, word2vec_embeding])
#     print("生成的数据特征形状:", features.shape)

#     # dom分类模型加载
#     model = DOM()
#     # model.load_models()

#     # 类别预测
#     pred = model.predict_voting(features)
#     processed_data['dom'] = pred
#     processed_data.drop('tokenized',axis = 1, inplace = True)

#     # ======================= 关键词判断类别 进行修正 ====================================
#     from DOM.ML.utils import based_on_keywords

#     key_df = based_on_keywords(processed_data)
#     key_df.loc[key_df['dom_key']==-1,'dom_key'] = processed_data[key_df['dom_key']==-1]['dom']

#     # 查看预测和关键词判断不同的数据
#     print("\n查看预测和关键词判断不同的数据：")
#     temp = key_df[key_df['dom']!=key_df['dom_key']]
#     print(temp[['text','dom','dom_key']])

#     key_df['dom'] = key_df['dom_key']
#     key_df.drop('dom_key',axis=1,inplace=True)

#     # 保存结果
#     key_df.to_csv(save_path, sep="\t", index=False)

if __name__ == "__main__":

    recognized_path = "/mnt/disk/wjh23/EaseDine/ASR/FunASR/A_audio_results/FunASR__enhenced_cer_0_over_0.txt"
    save_path = "/mnt/disk/wjh23/EaseDine/DOM/A_audio_results/A_audio_recognition_dom_4_27.txt"
    dom = DOM()
    dom.pre_dom(recognized_path, save_path)