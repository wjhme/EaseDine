# 核心库导入
import os
import torch
import shutil
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# FunASR相关导入
from funasr import AutoModel


def check_environment():
    """环境依赖检查"""
    required = {
        'funasr': '0.8.7',
        'torch': '2.1.0',
        'torchaudio': '2.1.0'
    }

    print("=" * 40)
    print("运行环境验证：")
    for pkg, ver in required.items():
        try:
            current = __import__(pkg).__version__
            status = "✓" if current == ver else f"× 需要 {ver}"
            print(f"{pkg:12} {current:10} {status}")
        except:
            print(f"{pkg:12} 未安装")
    print("=" * 40)
    print("\n")


def clean_model_cache():
    """深度清理模型缓存"""
    cache_paths = [
        os.path.expanduser('~/.cache/funasr'),
        os.path.expanduser('~/.cache/modelscope'),
        os.path.expanduser('~/.cache/torch')
    ]
    for path in cache_paths:
        if os.path.exists(path):
            print(f"清理缓存目录: {path}")
            shutil.rmtree(path, ignore_errors=True)


def load_audio(audio_path, target_sr=16000):
    """音频加载与预处理"""
    try:
        audio_file = Path(audio_path).absolute()
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")

        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_file)
        temp_file = None

        # 采样率转换
        if sample_rate != target_sr:
            print(f"采样率转换: {sample_rate}Hz -> {target_sr}Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sr
            )
            waveform = resampler(waveform)

            # 保存临时文件
            temp_file = audio_file.parent / f"temp_{target_sz}_{audio_file.name}"
            torchaudio.save(temp_file, waveform, target_sr)
            print(f"生成临时文件: {temp_file}")
            return str(temp_file), True

        return str(audio_file), False

    except Exception as e:
        print(f"音频处理失败: {str(e)}")
        raise


def asr_pipeline(audio_path):
    """语音识别流程"""
    try:
        # 初始化模型（自动下载）
        model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            model_revision="v1.2.4",
            vad_revision="v2.0.4",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 执行识别
        result = model.generate(audio_path)

        # 解析结果
        if len(result) > 0 and "text" in result[0]:
            return result[0]["text"]
        return "未识别到有效内容"

    except Exception as e:
        print(f"识别过程中发生错误: {str(e)}")
        raise


def main_process(audio_file):
    """主处理流程"""
    temp_created = False
    processed_path = None

    try:
        # 环境检查
        check_environment()

        # 清理旧缓存
        clean_model_cache()

        # 处理音频
        processed_path, temp_created = load_audio(audio_file)
        print(f"\n处理音频路径: {processed_path}")

        # 显示音频
        waveform, sr = torchaudio.load(processed_path)
        display(Audio(waveform, rate=sr))
        plot_waveform(waveform, sr)

        # 执行识别
        print("\n开始语音识别...")
        result_text = asr_pipeline(processed_path)
        print("\n" + "=" * 40)
        print(f"最终识别结果: {result_text}")
        print("=" * 40)

    except Exception as e:
        print(f"\n流程异常终止: {str(e)}")
    finally:
        # 清理临时文件
        if temp_created and processed_path:
            os.remove(processed_path)
            print(f"\n已清理临时文件: {processed_path}")


def plot_waveform(waveform, sample_rate):
    """可视化波形"""
    plt.figure(figsize=(12, 3))
    plt.plot(waveform[0].numpy())
    plt.title("音频波形")
    plt.xlabel(f"采样点 (总长度: {len(waveform[0]) / sample_rate:.2f}s)")
    plt.ylabel("振幅")
    plt.show()


if __name__ == "__main__":
    # 配置参数
    audio_file = "test1.wav"  # 支持wav/mp3格式

    # 启动处理流程
    main_process(audio_file)