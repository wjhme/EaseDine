from pydub import AudioSegment

def split_audio_middle(input_path, output_prefix="/mnt/disk/wjh23/EaseDine/ASR/data_process/temp_split_audio/"):
    # 1. 加载音频文件
    audio = AudioSegment.from_file(input_path)
    duration_ms = len(audio)  # 音频总时长（毫秒）
    
    # 2. 检查音频是否超过10秒
    if duration_ms <= 10 * 1000:
        print("音频长度 ≤10秒，无需切割")
        return
    
    # 3. 计算中间切割点（毫秒）
    split_point = duration_ms // 2  # 取中间点
    
    # 4. 切割成两段
    first_half = audio[:split_point]
    second_half = audio[split_point:]
    
    # 5. 保存结果
    first_half.export(f"{output_prefix}1.wav", format="wav")
    second_half.export(f"{output_prefix}2.wav", format="wav")
    print(f"音频已从中间切割为两段：{split_point/1000:.2f}秒处")

# 示例调用
split_audio_middle("/mnt/disk/wjh23/EaseDineDatasets/dataset_b/9e6f47da-c596-4b25-bad0-19e459edb008.wav")