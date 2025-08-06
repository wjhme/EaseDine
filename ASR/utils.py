from pathlib import Path
import Levenshtein
import pandas as pd
import os
import re



def calculate_cer(reference: str, hypothesis: str, 
                 detail_analysis: bool = False) -> float:
    """
    计算中文字错率（CER）
    
    参数:
        reference : 标注文本（标准答案）
        hypothesis : 识别结果文本
        detail_analysis : 是否返回详细编辑操作统计
        
    返回:
        CER值（当detail=True时返回包含详细信息的字典）
    """
    
    edit_ops = Levenshtein.editops(reference, hypothesis)
    edits = {'insertions': 0, 'deletions': 0, 'substitutions': 0}
    for op in edit_ops:
        if op[0] == 'insert':
            edits['insertions'] += 1
        elif op[0] == 'delete':
            edits['deletions'] += 1
        elif op[0] == 'replace':
            edits['substitutions'] += 1
    distance = Levenshtein.distance(reference, hypothesis)
    
    # 计算总字符数（中文按字统计）
    ref_len = len(reference)
    
    # 处理空标注文本的特殊情况
    if ref_len == 0:
        cer = 1.0 if len(hypothesis) > 0 else 0.0
        if detail_analysis:
            return {'cer': cer, 'edits': edits, 'distance': distance}
        return cer
    
    cer = distance / ref_len
    
    if detail_analysis:
        return {
            'cer': cer,
            'distance': distance,
            'ref_len': ref_len,
            'edits': edits,
            'insertions': edits['insertions'],
            'deletions': edits['deletions'],
            'substitutions': edits['substitutions']
        }
    return cer

def load_data(transcriptions_file, train_audio):
    """
    加载训练数据
    :param transcriptions_file: 标注文件路径
    :param train_audio: 包含所有音频子文件夹的根目录路径
    :return: [(uttid, wav_path, text), ...]
    """
    data = []
    
    # 读取标注文件内容，构建 {uttid: text} 的映射
    uttid_to_text = {}
    with open(transcriptions_file, "r", encoding="utf-8") as f:
        for line in f:
            uttid, text = line.strip().split("\t", 1)
            uttid_to_text[uttid] = text
    
    # 动态读取 train_audio 下的所有子文件夹
    train_audio_root = Path(train_audio)  # 转换为 Path 对象
    audio_dirs = [subdir for subdir in train_audio_root.iterdir() if subdir.is_dir()]

    # 遍历所有音频文件夹，查找对应的标注
    for audio_dir in audio_dirs:
        audio_dir = Path(audio_dir)  # 转换为 Path 对象
        for wav_file in audio_dir.glob("*.wav"):  # 遍历音频文件夹中的 .wav 文件
            uttid = wav_file.stem  # 获取文件名（去掉扩展名）
            if uttid in uttid_to_text:  # 检查是否在标注文件中有对应文本
                text = uttid_to_text[uttid]
                data.append((uttid, str(wav_file), text))  # 添加到数据列表
            else:
                print(f"警告：未找到 {uttid} 的标注信息，跳过该文件。")
    
    return data

# 加载测试数据，生成语音文件代号和语音文件路径列表
def load_pred_data(pred_audio):
    batch_uttid = []
    batch_wav_path = []
    audio_dir = Path(pred_audio)  # 转换为 Path 对象
    for wav_file in audio_dir.glob("*.wav"):  # 遍历音频文件夹中的 .wav 文件
        uttid = wav_file.stem  # 获取文件名（去掉扩展名）
        batch_uttid.append(uttid)
        batch_wav_path.append(str(wav_file))

    return batch_uttid, batch_wav_path

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

    # # 定义替换规则字典（左边为需要替换的词，右边为目标词）
    # replacement_rules = {
    #     '要往': '要碗'
    # }

    # # 生成正则表达式模式（按关键词长度降序排列，避免短词优先匹配）
    # patterns = sorted(replacement_rules.keys(), key=len, reverse=True)
    # regex_pattern = re.compile('|'.join(map(re.escape, patterns)))

    # data_df['text'] = data_df['text'].str.replace(regex_pattern, lambda x: replacement_rules[x.group()], regex=True)

    # 删除空格、标点符号
    data_df["text"] = data_df["text"].str.replace(r'[\s，。：“”？！,.?!]', '', regex=True)
    # 将小写字母转为大写
    data_df["text"] = data_df["text"].str.upper()

    # 保存识别结果
    data_df.to_csv(output_file, sep="\t", index=False, header=None)

def get_folderames(path):

    # 获取所有子文件夹名称（非递归）
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    # print("直接子文件夹:", subdirs)

    # # 获取所有子文件夹名称（递归）
    # all_subdirs = []
    # for root, dirs, files in os.walk(path):
    #     all_subdirs.extend(dirs)
    # print("所有子文件夹（递归）:", all_subdirs)
    return subdirs

def get_files(directory):
    """
    统计文件夹及其子文件夹中的文件数量
    :param directory: 目标文件夹路径
    :return: 文件总数（包含子文件夹）, 文件名列表
    """
    total = 0
    filenams_ls = []
    try:
        # 遍历文件夹树
        for root, dirs, filenams_ls in os.walk(directory):
            total += len(filenams_ls)
        return total, filenams_ls
    except Exception as e:
        print(f"错误：{e}")
        return -1

def merge_txt_files_to_dataframe(file_list, base_path=None):
    """
    合并多个文本文件到一个Pandas数据框
    
    参数：
        file_list: 文件名列表（如 ['train_audio_batch_1.txt', ...]）
        base_path: 文件所在基础路径
        
    返回：
        合并后的DataFrame
    """
    dfs = []
    
    for filename in file_list:
        filepath = Path(base_path) / filename
        try:
            # 读取时处理可能的编码问题（指定utf-8或自动检测）
            df = pd.read_csv(filepath, 
                            sep='\t',  # 假设是制表符分隔，按实际修改
                            encoding='utf-8',
                            on_bad_lines='warn')  # 跳过错误行
            # 生成 uuid_path 列
            dir_path = '/mnt/disk/wjh23/EaseDineDatasets/train_audio/'  # 可以修改为您需要的实际路径
            df['uuid_path'] = dir_path + filename[:-8] + '/' + df['uuid'] + '.wav'

            dfs.append(df)
            
        except Exception as e:
            print(f"警告：文件 {filename} 读取失败 - {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("所有文件读取失败，请检查路径和文件格式")
    
    # 合并时处理内存优化
    merged_df = pd.concat(dfs, 
                         axis=0, 
                         ignore_index=True,
                         copy=False)  # 减少内存复制
    
    # 后处理：去重和重置索引
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)
    print(f"总共有 {merged_df.shape[0]} 条数据.")
    
    return merged_df

if __name__=="__main__":
    results = pd.read_csv("/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/A_audio_results/FireRed_A_audio_beam_size_5.txt", sep="\t", header=None, names=["uttid", "text"])
    results_ls = results.to_dict('records')
    save_results_to_txt(results_ls, "/mnt/disk/wjh23/EaseDine/ASR/FireRedASR/A_audio_results/FireRed_A_audio_beam_size_5_new.txt")