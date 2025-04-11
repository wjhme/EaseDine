from pathlib import Path
# from funasr import AutoModel
from FunASRProcessor import FunASRProcessor
import torch


class FunASRFineTuner(FunASRProcessor):
    """增强版模型微调器（集成实战经验）"""

    def __init__(self,
                 train_data: str,
                 valid_data: str = None,
                 model_name: str = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                 batch_size: int = None,
                 **kwargs):
        """
        改进点：
        - 自动检测并生成数据清单文件
        - 智能batch_size设置
        - 集成文本清洗功能
        """
        super().__init__(model_name=model_name, **kwargs)

        # 数据路径处理
        self.train_data = self._validate_data_path(train_data, "训练")
        self.valid_data = self._validate_data_path(valid_data, "验证") if valid_data else None

        # 自动配置训练参数
        self.batch_size = self._auto_detect_batch_size() if batch_size is None else batch_size
        self.num_epochs = 10
        self.learning_rate = 1e-4

        # 训练环境自检
        self._environment_check()

    def _validate_data_path(self, data_path: str, data_type: str) -> Path:
        """数据路径验证与格式检查"""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"{data_type}数据路径不存在: {path}")

        # 自动生成数据清单文件（如果不存在）
        if not (path / "wav.scp").exists():
            self._generate_scp_file(path)
        if not (path / "text.txt").exists():
            self._generate_text_file(path)

        return path

    def _generate_scp_file(self, data_dir: Path):
        """自动生成wav.scp文件"""
        print(f"生成音频清单文件: {data_dir / 'wav.scp'}")
        audio_files = list(data_dir.glob("*.wav")) + list(data_dir.glob("*.mp3"))

        with open(data_dir / "wav.scp", "w") as f:
            for idx, audio in enumerate(audio_files, 1):
                utt_id = f"utt_{idx:05d}"  # 生成5位数字编号
                f.write(f"{utt_id} {audio.absolute()}\n")

    def _generate_text_file(self, data_dir: Path):
        """自动生成text.txt文件（需存在原始标注）"""
        raw_text = data_dir / "transcript.txt"
        if not raw_text.exists():
            raise FileNotFoundError(f"未找到原始标注文件: {raw_text}")

        print(f"生成清洗后的文本文件: {data_dir / 'text.txt'}")
        with open(raw_text) as fin, open(data_dir / "text.txt", "w") as fout:
            for line in fin:
                # 文本清洗：去除非中文字符
                cleaned = self._clean_text(line.strip())
                fout.write(f"{cleaned}\n")

    def _clean_text(self, text: str) -> str:
        """文本清洗（处理实战中的特殊字符问题）"""
        # 保留中文、数字、常用标点
        import re
        cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。？、！]", "", text)
        return cleaned.strip()

    def _auto_detect_batch_size(self) -> int:
        """智能batch_size检测（根据显存容量）"""
        if self.device == "cpu":
            return 16  # CPU模式默认批次

        # 检测可用显存
        torch.cuda.empty_cache()
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)

        # 预估模型内存需求（根据实战经验）
        mem_per_sample = 320 * 1024 * 1024  # 320MB/sample (16kHz音频)
        safe_batch_size = int(free_mem / mem_per_sample * 0.8)  # 保留20%余量

        print(f"自动设置batch_size={safe_batch_size} (可用显存:{free_mem // 1024 ** 2}MB)")
        return max(safe_batch_size, 2)  # 最小批次为2

    def _environment_check(self):
        """环境兼容性检查"""
        cuda_available = torch.cuda.is_available()
        print("\n" + "=" * 40)
        print("环境自检报告：")
        print(f"- PyTorch版本: {torch.__version__}")
        print(f"- CUDA可用: {cuda_available}")
        if cuda_available:
            print(f"- CUDA版本: {torch.version.cuda}")
            print(f"- 当前GPU: {torch.cuda.get_device_name(0)}")

        # 显存警告
        if self.device == "cuda" and self.batch_size > 32:
            print("\n警告：batch_size超过32可能导致显存不足！")
            print("建议：降低batch_size或使用混合精度训练")

    def prepare_config(self):
        """生成配置文件（整合实战参数）"""
        config = {
            "train": {
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "train_data": str(self.train_data.absolute()),
                "valid_data": str(self.valid_data.absolute()) if self.valid_data else None
            },
            "model": {
                "pretrained_model": self.model_name,
                "freeze_encoder": False  # 实战经验：不解码预训练层
            }
        }
        return config

    def train(self, resume: bool = False):
        """增强训练流程"""
        try:
            # 生成数据配置
            data_config = {
                "train": {
                    "scp": str(self.train_data / "wav.scp"),
                    "text": str(self.train_data / "text.txt")
                }
            }
            if self.valid_data:
                data_config["valid"] = {
                    "scp": str(self.valid_data / "wav.scp"),
                    "text": str(self.valid_data / "text.txt")
                }

            # 调用FunASR训练API
            self.model.train(
                data_config=data_config,
                output_dir="checkpoints",
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                lr=self.learning_rate,
                resume=resume
            )

            # 训练后评估
            if self.valid_data:
                self._evaluate()

        except Exception as e:
            print(f"\n训练中断: {str(e)}")
            print("常见问题排查：")
            print("1. 检查数据路径是否正确（音频文件是否存在）")
            print("2. 尝试降低batch_size解决显存不足问题")
            print("3. 检查文本文件是否包含非法字符")
            raise

    def _evaluate(self):
        """在验证集上评估模型"""
        print("\n开始验证集评估...")
        results = []
        with open(self.valid_data / "text.txt") as f:
            references = [line.strip() for line in f]

        # 执行批量推理
        with torch.no_grad():
            for audio_path in self.valid_data.glob("*.wav"):
                result = self.transcribe(str(audio_path))
                results.append(result["text"])

        # 计算词错误率（WER）
        from cer_evaluate import calculate_cer
        cer_scores = []
        for i in range(len(references)):
            cer_score = calculate_cer(references[i], results[i])
            cer_scores.append(cer_score)
        print(f"验证集CER: {(1 - sum(cer_scores)/len(cer_scores)) * 100:.2f}%")

    def save_model(self, save_path: str):
        """保存微调后的模型"""
        self.model.save_pretrained(save_path)
        print(f"模型已保存至: {save_path}")