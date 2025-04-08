from pathlib import Path

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
            text.replace(" ","")
            # 写入格式：uttid 文本标注
            f.write(f"{uttid}\t{text}\n")
    print(f"结果已保存到 {output_file}")
