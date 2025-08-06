import os
import time
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from functools import partial
from noisereduce import reduce_noise
from multiprocessing import Pool, cpu_count

def process_single_file(input_path, output_dir, target_rms=0.03):
    """降噪处理单个音频文件"""
    try:
        output_path = Path(output_dir) / f"cleaned_{Path(input_path).name}"
        y, sr = librosa.load(input_path, sr=16000, mono=True)
        
        audio_norm = librosa.util.normalize(y)  # 峰值归一化
        
        reduced_noise = reduce_noise(
            y=audio_norm,
            sr=sr,
            stationary=False,
            prop_decrease=0.8,
            n_fft=256,
            win_length=256,
            hop_length=128,
            n_jobs=1
        )
        
        # rms = np.sqrt(np.mean(reduced_noise**2))
        # gain = target_rms / (rms + 1e-8)
        # y_boosted = reduced_noise * np.clip(gain, 1.0, 5.0)
        
        # enhanced_audio = librosa.effects.preemphasis(y_boosted, coef=0.97)
        sf.write(output_path, reduced_noise, sr, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"处理失败: {input_path} - {str(e)}")
        return False

def batch_process_audio(input_dir, output_dir, num_workers=None):
    """
    批量处理音频文件
    
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    :param num_workers: 并行进程数 (默认使用全部核心)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}
    input_files = [
        str(p) for p in Path(input_dir).iterdir() 
        if p.suffix.lower() in audio_exts
    ]
    
    num_workers = num_workers or min(cpu_count(), 10)
    print(f"启动 {num_workers} 个进程处理 {len(input_files)} 个文件...")
    
    # 使用 partial 固定参数，避免 lambda
    process_func = partial(
        process_single_file,
        output_dir=output_dir,
        target_rms=0.03
    )
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_func, input_files),
            total=len(input_files),
            desc="处理进度"
        ))
    
    success_count = sum(results)
    print(f"处理完成! 成功: {success_count}, 失败: {len(input_files)-success_count}")

import os
import torch
import librosa
import soundfile as sf
from spleeter.separator import Separator

# 音频分离
def separate_vocals(input_file, output_dir):
    # 使用 Spleeter 作为示例后端，Spleeter 可以实现音频分离
    separator = Separator('spleeter:2stems')
    # 执行分离操作
    prediction = separator.separate_to_file(input_file, output_dir)

    # 示例路径，假设分离后的人声和伴奏文件
    vocals_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0], 'vocals.wav')
    accompaniment_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0], 'accompaniment.wav')

    return vocals_path, accompaniment_path


import librosa
import numpy as np

def audio_quality_check(audio_path, sample_rate=16000, low_thresh_db=-30, high_thresh_db=-3, clip_thresh=0.99):
    """
    音频音量与失真检测
    
    参数:
        audio_path: 音频文件路径
        sample_rate: 采样率(默认16000)
        low_thresh_db: 音量过低阈值(dB, 默认-30)
        high_thresh_db: 音量过高阈值(dB, 默认-3)
        clip_thresh: 削波检测阈值(0-1, 默认0.99表示16-bit音频中绝对值≥32767*0.99)
    
    返回:
        dict: 包含检测结果和问题片段的字典
    """
    # 加载音频文件
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # 如果是立体声，取平均值转为单声道
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
    
    # 初始化结果字典
    result = {
        'problems': [],
        'low_volume_segments': [],
        'high_volume_segments': [],
        'clipping_segments': [],
        'max_amplitude': float(np.max(np.abs(audio))),
        'rms': float(librosa.feature.rms(y=audio)[0].mean()),
        'sample_rate': sr,
        'duration': librosa.get_duration(y=audio, sr=sr)
    }
    
    # 计算分帧参数 (每帧100ms)
    frame_length = int(sr * 0.1)
    hop_length = frame_length // 2
    
    # 分帧处理
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    
    # 计算每帧的RMS和dB值
    rms_values = []
    db_values = []
    for i in range(frames.shape[1]):
        frame = frames[:, i]
        rms = np.sqrt(np.mean(frame**2))
        rms_values.append(rms)
        db = 20 * np.log10(rms + 1e-10)  # 避免log(0)
        db_values.append(db)
    
    # 检测音量问题
    for i, db in enumerate(db_values):
        start_time = i * hop_length / sr
        end_time = start_time + frame_length / sr
        
        # 检测低音量
        if db < low_thresh_db:
            result['low_volume_segments'].append((start_time, end_time, db))
        
        # 检测高音量
        if db > high_thresh_db:
            result['high_volume_segments'].append((start_time, end_time, db))
    
    # 检测削波
    clip_thresh_abs = clip_thresh * (2**15 - 1) if audio.dtype == np.int16 else clip_thresh
    clip_samples = np.where(np.abs(audio) >= clip_thresh_abs)[0]
    
    if len(clip_samples) > 0:
        # 将连续的削波样本分组为片段
        clip_segments = []
        start = clip_samples[0]
        prev = start
        
        for sample in clip_samples[1:]:
            if sample > prev + 1:  # 不连续
                clip_segments.append((start/sr, prev/sr))
                start = sample
            prev = sample
        clip_segments.append((start/sr, prev/sr))
        
        result['clipping_segments'] = clip_segments
    
    # 汇总问题
    if result['low_volume_segments']:
        result['problems'].append(f"发现低音量(<{low_thresh_db}dB)片段")
    if result['high_volume_segments']:
        result['problems'].append(f"发现高音量(>{high_thresh_db}dB)片段")
    if result['clipping_segments']:
        result['problems'].append("发现削波片段")
    if not result['problems']:
        result['problems'].append("音频质量良好，未发现问题")
    
    return result


import os
import csv
import random
import torchaudio
from audiomentations import Compose, AddBackgroundNoise, TimeStretch, PitchShift, Gain
from multiprocessing import Pool
from tqdm import tqdm

# 配置增强参数
class AugConfig:
    # 输入输出路径
    scp_path = "/mnt/disk/wjh23/FunASR/finetuner/train_data/scp_txt/batch_all.scp"
    output_audio_dir = "/mnt/disk/wjh23/EaseDineDatasets/processed_audio/enhanced_audio_batch_all"
    original_text_path = "/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/train.txt"
    output_text_path = "/mnt/disk/wjh23/EaseDineDatasets/智慧养老_label/enhanced_audio_batch_all.txt"
    noise_dir = "/mnt/disk/wjh23/EaseDineDatasets/musan/sound-bible"

    # 增强配置列表（每个增强包含实例和独立概率）
    augmenters = [
        {  # 语速调整
            "augmenter": TimeStretch(
                min_rate=0.8, max_rate=1.5,
                p=1.0  # p设为1，概率由外层控制
            ),
            "prob": 0.6  # 原p=0.6
        },
        {  # 音高偏移
            "augmenter": PitchShift(
                min_semitones=-4, max_semitones=4,
                p=1.0
            ),
            "prob": 0.6
        },
        {  # 背景噪声
            "augmenter": AddBackgroundNoise(
                noise_dir,
                min_snr_db=5, max_snr_db=15,
                p=1.0
            ),
            "prob": 0.5
        },
        {  # 音量调整
            "augmenter": Gain(
                min_gain_db=-7, max_gain_db=7,
                p=1.0
            ),
            "prob": 0.5
        }
    ]


# 加载SCP文件
def load_scp_file():
    audio_map = {}
    with open(AugConfig.scp_path) as f:
        for line in f:
            uuid, path = line.strip().split()
            audio_map[uuid] = path
    return audio_map

# 加载原始标注
def load_text_annotations():
    text_map = {}
    with open(AugConfig.original_text_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                uuid, text, dom = row[0], row[1], row[2]
                text_map[uuid] = (text, dom)
    return text_map

# 音频增强处理
def process_audio(args):
    uuid, audio_path = args
    try:
        # 加载原始音频
        waveform, orig_sr = torchaudio.load(audio_path)
        waveform = waveform.numpy()[0]

        # 统一采样率
        if orig_sr != 16000:
            waveform = torchaudio.functional.resample(
                torch.tensor(waveform), orig_sr, 16000
            ).numpy()

        # 应用增强（新逻辑）
        augmented = waveform.copy()
        applied = 0
        
        # 随机打乱增强顺序
        shuffled_augmenters = random.sample(AugConfig.augmenters, len(AugConfig.augmenters))
        
        for aug_info in shuffled_augmenters:
            if applied >= 2:
                break
            # 根据概率决定是否应用该增强
            if random.random() < aug_info["prob"]:
                augmented = aug_info["augmenter"](augmented, sample_rate=16000)
                applied += 1
        
        # 保存增强音频
        output_path = os.path.join(
            AugConfig.output_audio_dir, 
            f"{uuid}_aug.wav"
        )
        torchaudio.save(output_path, torch.tensor([augmented]), 16000)
        
        return uuid, output_path
    
    except Exception as e:
        print(f"Error processing {uuid}: {str(e)}")
        return None

# 批量处理主函数
def batch_augmentation(num_workers=8):
    # 创建输出目录
    os.makedirs(AugConfig.output_audio_dir, exist_ok=True)
    
    # 加载数据
    audio_map = load_scp_file()
    text_map = load_text_annotations()
    
    # 准备处理队列
    tasks = []
    for uuid, path in audio_map.items():
        if uuid in text_map:  # 只处理有标注的音频
            tasks.append((uuid, path))
    
    # 多进程处理
    augmented_data = []
    with Pool(num_workers) as pool:
        results = pool.imap(process_audio, tasks)
        for result in tqdm(results, total=len(tasks)):
            if result:
                uuid, audio_path = result
                original_text, dom = text_map[uuid]
                # 文本适配
                # augmented_text = adapt_dialect(original_text)
                augmented_data.append({
                    "uuid": uuid,
                    "audio_path": audio_path,
                    "text": original_text,
                    # "augmented_text": augmented_text,
                    "dom": dom
                })
    
    # 保存增强后的标注文件
    with open(AugConfig.output_text_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        for item in augmented_data:
            writer.writerow([
                item["uuid"] + "_aug",
                # item["augmented_text"],
                item["text"],
                item["dom"]
                # os.path.basename(item["audio_path"])
            ])

if __name__ == "__main__":
    # 启动增强流程
    batch_augmentation(num_workers=20)
    print(f"处理完成！增强数据保存在：{AugConfig.output_audio_dir}")
    print(f"新标注文件：{AugConfig.output_text_path}")


# if __name__ == "__main__":
#     # =================== 降噪处理示例用法 ===================
#     input_directory = "/mnt/disk/wjh23/EaseDineDatasets/train_audio/train_audio_batch_1"
#     output_directory = "/mnt/disk/wjh23/EaseDineDatasets/processed_audio/train_audio_batch_1"
    
#     # 执行批量处理
#     # t0 = time.time()
#     # batch_process_audio(input_directory, output_directory)
#     # print(f"音频降噪并归一化处理用时：{time.time() - t0:.2f} s")

#     input_path = "/mnt/disk/wjh23/EaseDineDatasets/热词音频/0a0c93c1-2b5c-42dc-aefa-6e238aad5051.wav"
#     output_dir = "/mnt/disk/wjh23/EaseDineDatasets/音频降噪后语音"
#     process_single_file(input_path, output_dir)

    # =================== 人声分离示例 ===================
    # input_file = "/mnt/disk/wjh23/EaseDineDatasets/存在噪音的语音/4d1826a9-2821-48e9-af58-d3136b796d71.wav"  # 替换为实际的音频文件路径
    # output_dir = "separated_audio"  # 替换为实际的输出目录

    # vocals_path, accompaniment_path = separate_vocals(input_file, output_dir)
    # print(f"人声文件保存路径: {vocals_path}")
    # print(f"伴奏文件保存路径: {accompaniment_path}")

    # =================== 音量检测示例 ===================
    #     # 替换为你的音频文件路径
    # audio_file = "/mnt/disk/wjh23/EaseDineDatasets/存在噪音的语音/83903212-685f-4caf-8dae-557bcec1619a.wav"
    
    # # 检测音频质量
    # result = audio_quality_check(audio_file)
    
    # print("\n音频质量检测结果:")
    # print(f"采样率: {result['sample_rate']}Hz")
    # print(f"时长: {result['duration']:.2f}秒")
    # print(f"最大振幅: {result['max_amplitude']:.4f}")
    # print(f"平均RMS: {result['rms']:.4f}")
    
    # print("\n检测到的问题:")
    # for problem in result['problems']:
    #     print(f"- {problem}")
    
    # if result['low_volume_segments']:
    #     print(f"\n低音量片段(<-30dB): {len(result['low_volume_segments'])}处")
    #     for start, end, db in result['low_volume_segments'][:3]:  # 只显示前3处
    #         print(f"  {start:.2f}-{end:.2f}s ({db:.1f}dB)")
    
    # if result['high_volume_segments']:
    #     print(f"\n高音量片段(>-3dB): {len(result['high_volume_segments'])}处")
    #     for start, end, db in result['high_volume_segments'][:3]:
    #         print(f"  {start:.2f}-{end:.2f}s ({db:.1f}dB)")
    
    # if result['clipping_segments']:
    #     print(f"\n削波片段: {len(result['clipping_segments'])}处")
    #     for start, end in result['clipping_segments'][:3]:
    #         print(f"  {start:.4f}-{end:.4f}s")





