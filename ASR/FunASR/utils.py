import pandas as pd
import re

def save_results_to_txt(results, output_file, sort=True):
    """
    将 results (uuid, text, dom) 保存为 txt 文件(按官方uuid顺序调整)
    :param results: 包含识别结果的列表，每个元素是一个字典
    :param output_file: 输出文件路径
    """
    data_df = results
    # 从uuid列提取语音编号
    data_df['uuid_temp'] = data_df['uuid'].str.extract(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})')

    # 更新uuid列并删除临时列
    data_df['uuid'] = data_df['uuid_temp']
    data_df.drop(['uuid_temp', 'status', 'time'], axis=1, inplace=True)

    # 定义替换规则字典（左边为需要替换的词，右边为目标词）
    replacement_rules = {
        "\ufffd": "",    # Unicode 替换字符 �
        '要往': '要碗',
        '来问': '来碗',
        # '来玩': '来碗',
        '小草': '小炒'
    }

    # 生成正则表达式模式（按关键词长度降序排列，避免短词优先匹配）
    patterns = sorted(replacement_rules.keys(), key=len, reverse=True)
    regex_pattern = re.compile('|'.join(map(re.escape, patterns)))

    data_df['text'] = data_df['text'].str.replace(regex_pattern, lambda x: replacement_rules[x.group()], regex=True)

    # 删除空格、标点符号
    data_df["text"] = data_df["text"].str.replace(r'[\s，。？！,.?!]', '', regex=True)
    # 将小写字母转为大写
    data_df["text"] = data_df["text"].str.upper()
    # 只保留中文
    # data_df["text"] = data_df["text"].str.replace(r'[^\u4e00-\u9fa5]', '', regex=True) 

    # 处理uuid顺序
    # 官方提交文档
    if sort:
        A_df = pd.read_csv("/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/A.txt",sep="\t")[['uuid']]
        data_df = A_df.merge(data_df, on='uuid', how='left')

    # 识别为空的文本处理
    data_df['text'] = data_df['text'].fillna("天猫精灵")
    # 按比赛提交顺序保存识别结果
    data_df.to_csv(output_file, sep="\t", index=False, header=None)