# 核心库导入
import os
import torch
import time
import shutil
import torchaudio
from pathlib import Path
from IPython.display import Audio, display

# FunASR相关导入
from funasr import AutoModel

# 新增模型路径配置
MODEL_NAME = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
MODEL_REVISION = "v2.0.4"
MODEL_CACHE_DIR = os.path.expanduser("/mnt/disk/wjh23/EaseDine/ASR/model")  # 自定义模型缓存目录

# 设置模型缓存环境变量（在文件开头添加）
os.environ['MODELSCOPE_CACHE'] = MODEL_CACHE_DIR  # 强制指定缓存目录
os.environ['MODELSCOPE_HUB_CACHE'] = MODEL_CACHE_DIR
# os.environ['MODELSCOPE_FILE_LOCK'] = os.path.join(MODEL_CACHE_DIR, '.file_lock')

# 全局模型初始化（GPU优先）
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # 启用CUDA加速
t1 = time.time()
model = AutoModel(
    model=MODEL_NAME,
    model_revision=MODEL_REVISION,
    cache_dir=MODEL_CACHE_DIR,
    disable_update=True,
    device=device,
    punc_config={"enable": True},  # 启用标点预测
)
print(f"load model time:{time.time() - t1:.2f} s")

def load_audio(audio_path, target_sr=16000):
    """音频加载与预处理（直接返回张量）"""
    try:
        audio_file = Path(audio_path).resolve()
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")

        waveform, sample_rate = torchaudio.load(audio_file)

        # 采样率转换
        if sample_rate != target_sr:
            print(f"采样率转换: {sample_rate}Hz -> {target_sr}Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sr
            )
            waveform = resampler(waveform)
            sample_rate = target_sr

        # 确保单声道，兼容模型输入
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, sample_rate

    except Exception as e:
        print(f"音频处理失败: {str(e)}")
        raise

def asr_pipeline(waveform, sample_rate):
    """语音识别流程（直接处理音频张量）"""
    try:
        # 转换为模型需要的格式（numpy数组）
        waveform_np = waveform.squeeze().numpy()  # 去除通道维度
        
        # 执行识别（假设模型支持numpy输入）
        t1 = time.time()
        result = model.generate(input=waveform_np, sample_rate=sample_rate)
        print(f"执行识别：{time.time() - t1:.2f} s")

        if len(result) > 0 and "text" in result[0]:
            return result[0]["text"]
        return "未识别到有效内容"

    except Exception as e:
        print(f"识别过程中发生错误: {str(e)}")
        raise

def main_process(audio_file):
    """主处理流程"""
    try:
        # 加载并预处理音频
        waveform, sample_rate = load_audio(audio_file)
        print("\n音频张量形状:", waveform.shape)

        # # 显示音频
        # display(Audio(waveform.numpy(), rate=sample_rate))

        # 执行识别
        print("\n开始语音识别...")
        t0 = time.time()
        result_text = asr_pipeline(waveform, sample_rate)
        spend_time = time.time() - t0
        
        print("\n" + "=" * 40)
        print(f"最终识别结果: {result_text}\n 总用时：{spend_time:.2f} s")
        print("=" * 40)

    except Exception as e:
        print(f"\n流程异常终止: {str(e)}")

def process_single_file(audio_file, display_detail=False):
    """单文件处理流程"""
    try:
        waveform, sample_rate = load_audio(audio_file)
        
        if display_detail:
            print("\n音频张量形状:", waveform.shape)
            display(Audio(waveform.numpy(), rate=sample_rate))

        t0 = time.time()
        result_text = asr_pipeline(waveform, sample_rate)
        spend_time = time.time() - t0
        
        return {
            "file": audio_file,
            "text": result_text,
            "duration": spend_time,
            "status": "success"
        }
    except Exception as e:
        print(f"处理失败: {audio_file}")
        return {
            "file": audio_file,
            "text": str(e),
            "duration": 0,
            "status": "failed"
        }

def batch_process(input_path, output_file="results.csv"):
    """批量处理主函数"""
    # 获取文件列表
    if os.path.isdir(input_path):
        audio_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
    elif os.path.isfile(input_path):
        audio_files = [input_path]
    else:
        raise ValueError("无效的输入路径")

    # 执行批量识别
    results = []
    total_count = len(audio_files)
    success_count = 0
    
    print(f"\n开始批量处理，共 {total_count} 个文件...")
    
    for idx, audio_file in enumerate(audio_files, 1):
        print(f"\n处理进度: {idx}/{total_count}")
        print("当前文件:", audio_file)
        
        result = process_single_file(audio_file)
        results.append(result)
        
        if result["status"] == "success":
            success_count += 1
            print(f"识别结果: {result['text']}")
        else:
            print(f"识别失败: {result['text']}")
        
        print(f"处理耗时: {result['duration']:.2f}s")
        print("-" * 60)

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("文件路径,识别结果,处理时间(秒),状态\n")
        for res in results:
            f.write(f"{res['file']},{res['text']},{res['duration']:.2f},{res['status']}\n")
    
    # 打印统计信息
    print("\n" + "="*60)
    print(f"批量处理完成！成功 {success_count}/{total_count}")
    print(f"详细结果已保存至: {output_file}")
    print("="*60)

if __name__ == "__main__":
    # 启动前预加载模型（可选）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # # 批处理
    # input_path = "../audios"  # 替换为你的音频文件/目录路径
    # batch_process(
    #     input_path=input_path,
    #     output_file="results/asr_results.csv"
    # )

    # 单文件处理
    input_path = "../audios/test2.wav"
    main_process(input_path)

    '''
    load model time:4.20 s
    
    开始语音识别...
    rtf_avg: 0.157: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.51it/s]
    执行识别：0.67 s

    ========================================
    最终识别结果: 甚至出现交易几乎停滞的情况
    总用时：0.67 s
    ========================================
    '''