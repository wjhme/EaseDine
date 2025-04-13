from pathlib import Path
import pandas as pd
import os
import re

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
    将 results 列表中的 uttid 和 text 保存为 txt 文件
    :param results: 包含识别结果的列表，每个元素是一个字典
    :param output_file: 输出文件路径
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            uttid = result["uttid"]
            text = result["text"]
            # 删除 空格、字母
            text = re.sub(r'[\sA-Za-z]', '', text)
            
            # 写入格式：uttid 文本标注
            f.write(f"{uttid}\t{text}\n")
    print(f"结果已保存到 {output_file}")

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

def merge_txt_files_to_dataframe(file_list, base_path="/mnt/disk/wjh23/FunASR/finetuner/train_data"):
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
            df['uuid_path'] = dir_path + filename[:-4] + '/' + df['uuid'] + '.wav'

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