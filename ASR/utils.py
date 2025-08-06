from pathlib import Path
import pandas as pd


def save_results_to_txt(results, output_file):
    """
    将 results (uuid, text, dom) 保存为 txt 文件(按官方uuid顺序调整)
    :param results: 包含识别结果的列表，每个元素是一个字典
    :param output_file: 输出文件路径
    """
    data_df = results.copy()
    # 从uuid列提取语音编号
    data_df['uuid_temp'] = data_df['uuid'].str.extract(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})')

    # 更新uuid列并删除临时列
    data_df['uuid'] = data_df['uuid_temp']
    data_df.drop(['uuid_temp', 'status', 'time'], axis=1, inplace=True)

    # 删除空格、标点符号
    data_df["text"] = data_df["text"].str.replace(r'[\s，。：“”？！,.?!]', '', regex=True)
    # 将小写字母转为大写
    data_df["text"] = data_df["text"].str.upper()

    # 保存识别结果
    data_df.to_csv(output_file, sep="\t", index=False, header=None)
