import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from functools import partial
from noisereduce import reduce_noise
from multiprocessing import Pool, cpu_count

# 将处理函数移到外层，确保可序列化
def process_single_file(input_path, output_dir, target_rms=0.03):
    """处理单个音频文件"""
    try:
        output_path = Path(output_dir) / f"cleaned_{Path(input_path).name}"
        y, sr = librosa.load(input_path, sr=16000, mono=True)
        
        reduced_noise = reduce_noise(
            y=y,
            sr=sr,
            stationary=False,
            prop_decrease=0.8,
            n_fft=256,
            win_length=256,
            hop_length=128,
            n_jobs=1
        )
        
        rms = np.sqrt(np.mean(reduced_noise**2))
        gain = target_rms / (rms + 1e-8)
        y_boosted = reduced_noise * np.clip(gain, 1.0, 5.0)
        
        enhanced_audio = librosa.effects.preemphasis(y_boosted, coef=0.97)
        sf.write(output_path, enhanced_audio, sr, subtype='PCM_16')
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

if __name__ == "__main__":
    # 示例用法
    input_directory = "/mnt/disk/wjh23/EaseDineDatasets/train_audio/train_audio_batch_1"
    output_directory = "/mnt/disk/wjh23/EaseDineDatasets/processed_audio/train_audio_batch_1"
    
    # 执行批量处理
    batch_process_audio(input_directory, output_directory)