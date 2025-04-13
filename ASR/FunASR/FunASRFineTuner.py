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
                uuid = f"{str(audio.absolute())[-40:-4]}"  
                f.write(f"{uuid} {audio.absolute()}\n")

    def _generate_text_file(self, data_dir: Path):
        """自动生成text.txt文件（需存在原始标注）"""
        raw_text = Path("/mnt/disk/wjh23/EaseDineDatasets/train_audio/transcript.txt")
        if not raw_text.exists():
            raise FileNotFoundError(f"未找到原始标注文件: {raw_text}")

        print(f"对应的文本文件: {data_dir / 'text.txt'}")
        import pandas as pd
        scp = pd.read_csv(data_dir / "wav.scp",sep=" ",names=['uuid','dir'])
        raw = pd.read_csv(raw_text,sep="\t", names=['uuid','text'])
        text = scp.merge(raw, on='uuid', how='left')[['uuid','text']]
        text.to_csv(data_dir / 'text.txt',sep=" ",index=False,header=None)

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

        print(f"- batch_size：{self.batch_size}")
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


# # funasr_finetune.py
# import os
# import torch
# import logging
# from pathlib import Path
# from omegaconf import OmegaConf
# from funasr import AutoModel
# from funasr.train_utils.trainer import Trainer

# class FunASRFinetuner:
#     """基于官方训练脚本的微调封装类"""
    
#     def __init__(self, 
#                  train_data: str,
#                  valid_data: str = None,
#                  pretrained_name: str = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
#                  output_dir: str = "./finetune_models"):
#         """
#         参数说明：
#         train_data: 训练数据目录路径（需含wav文件和transcript.txt）
#         valid_data: 验证数据目录路径（可选）
#         pretrained_name: 预训练模型名称或路径
#         output_dir: 微调输出目录
#         """
#         self.train_data = Path(train_data)
#         self.valid_data = Path(valid_data) if valid_data else None
#         self.pretrained_name = pretrained_name
#         self.output_dir = Path(output_dir)
        
#         # 环境初始化
#         self._setup_environment()
#         self._prepare_data_files()
        
#         # 自动配置参数
#         self.config = self._build_config()

#         print("数据集路径验证:")
#         print(f"训练路径: {self.train_data}")
#         print(f"SCP文件存在: {(self.train_data/'wav.scp').exists()}")
#         print(f"Text文件存在: {(self.train_data/'text.txt').exists()}")
#         print(f"音频文件数量: {len(list(self.train_data.glob('*.wav')))}\n")
        
#     def _setup_environment(self):
#         """初始化训练环境"""
#         torch.backends.cudnn.deterministic = True
#         self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # 创建输出目录
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#     def _prepare_data_files(self):
#         """准备训练所需的数据清单文件"""
#         for data_dir in [self.train_data, self.valid_data]:
#             if data_dir and data_dir.exists():
#                 self._generate_scp_file(data_dir)
#                 self._generate_text_file(data_dir)
                
#     def _generate_scp_file(self, data_dir: Path):
#         """生成wav.scp文件"""
#         scp_path = data_dir / "wav.scp"
#         if not scp_path.exists():
#             with open(scp_path, "w") as f:
#                 for wav_file in data_dir.glob("*.wav"):
#                     f.write(f"{wav_file.stem} {wav_file.name}\n")
                    
#     def _generate_text_file(self, data_dir: Path):
#         """生成text.txt文件"""
#         text_path = data_dir / "text.txt"
#         if not text_path.exists():
#             # 假设存在原始标注文件transcript.txt
#             raw_text = Path("/mnt/disk/wjh23/EaseDineDatasets/train_audio/transcript.txt")
#             if raw_text.exists():
#                 import pandas as pd
#                 scp = pd.read_csv(data_dir / "wav.scp",sep=" ",names=['uuid','dir'])
#                 raw = pd.read_csv(raw_text,sep="\t", names=['uuid','text'])
#                 text = scp.merge(raw, on='uuid', how='left')[['uuid','text']]
#                 text.to_csv(data_dir / 'text.txt',sep=" ",index=False,header=None)
#             else:
#                 raise FileNotFoundError(f"未找到原始标注文件: {raw_text}")

#     def _build_config(self) -> OmegaConf:
#         """构建Hydra配置"""
#         config = OmegaConf.create({
#             "model": self.pretrained_name,
#             "dataset_conf": {
#                 "train_data": str(self.train_data),
#                 "valid_data": str(self.valid_data) if self.valid_data else None,
#                 "tokenizer": None,  # 占位符，将在运行时填充
#                 "frontend": None,   # 占位符，将在运行时填充
#                 "batch_size": self._auto_batch_size(),
#                 "num_workers": 4
#             },
#             "optim": "adam",
#             "optim_conf": {
#                 "lr": 1e-4,
#                 "weight_decay": 0.0
#             },
#             "scheduler": "warmuplr",
#             "scheduler_conf": {
#                 "warmup_steps": 5000
#             },
#             "train_conf": {
#                 "output_dir": str(self.output_dir),
#                 "max_epochs": 10,
#                 "save_interval": 1,
#                 "use_fp16": torch.cuda.is_available()
#             }
#         })
#         return config
    
#     def _auto_batch_size(self) -> int:
#         """自动计算安全batch_size"""
#         if self.device == "cpu":
#             return 8
        
#         torch.cuda.empty_cache()
#         free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
#         return int(free_mem / (512 * 1024 * 1024))  # 按512MB/样本估算

#     def run(self):
#         """执行微调流程"""
#         # 初始化模型
#         model = AutoModel(
#             model=self.pretrained_name,
#             # 显式传递tokenizer和frontend参数
#             tokenizer_type="whisper_en" if "en" in self.pretrained_name else "zh",
#             frontend_type="fbank",
#             device=self.device
#         )
#         # 从model.kwargs获取相关组件
#         tokenizer = model.kwargs["tokenizer"]
#         frontend = model.kwargs["frontend"]
        
#         # 构建训练组件
#         train_components = self._build_train_components(model)
        
#         # 启动训练循环
#         self._train_loop(**train_components)

#     def _build_train_components(self, model):
#         """构建训练所需组件"""

#         # 获取必要组件
#         tokenizer = model.kwargs["tokenizer"]
#         frontend = model.kwargs["frontend"]

#         # 初始化数据集时传递这些组件
#         self.config.dataset_conf.update({
#             "tokenizer": tokenizer,
#             "frontend": frontend
#         })
#         # 模型参数冻结示例（冻结编码器）
#         for name, param in model.model.named_parameters():
#             if "encoder" in name:
#                 param.requires_grad = False

#         # 优化器
#         optimizer = torch.optim.AdamW(
#             filter(lambda p: p.requires_grad, model.model.parameters()),
#             **self.config.optim_conf
#         )

#         # 学习率调度器
#         scheduler = torch.optim.lr_scheduler.LambdaLR(
#             optimizer, 
#             lr_lambda=lambda step: min(step / 5000, 1.0)
#         )

#         # 训练器
#         trainer = Trainer(
#             local_rank=self.local_rank,
#             device=self.device,
#             **self.config.train_conf
#         )

#         return {
#             "model": model.model,
#             "optimizer": optimizer,
#             "scheduler": scheduler,
#             "trainer": trainer
#         }

#     def _train_loop(self, model, optimizer, scheduler, trainer):
#         """训练循环"""
#         # 数据加载（需根据实际数据集类实现）
#         from funasr.datasets.audio_datasets.datasets import  AudioDataset
        
#         # 从配置中获取组件
#         tokenizer = self.config.dataset_conf.tokenizer
#         frontend = self.config.dataset_conf.frontend
        
#         train_set = AudioDataset(
#             path=str(self.train_data),
#             scp_file="wav.scp",
#             text_file="text.txt",
#             tokenizer=tokenizer,
#             frontend=frontend
#         )
        
#         valid_set = AudioDataset(
#             scp_file=str(self.valid_data / "wav.scp"),
#             text_file=str(self.valid_data / "text.txt")
#         ) if self.valid_data else None

#         # 训练循环
#         for epoch in range(trainer.max_epochs):
#             # 训练阶段
#             trainer.train_epoch(
#                 model=model,
#                 optim=optimizer,
#                 scheduler=scheduler,
#                 dataloader_train=train_set,
#                 epoch=epoch
#             )
            
#             # 验证阶段
#             if valid_set:
#                 trainer.validate_epoch(
#                     model=model,
#                     dataloader_val=valid_set,
#                     epoch=epoch
#                 )
            
#             # 保存检查点
#             if self.local_rank == 0:
#                 trainer.save_checkpoint(
#                     epoch + 1,
#                     model=model,
#                     optim=optimizer,
#                     scheduler=scheduler
#                 )

# if __name__ == "__main__":
#     # 使用示例
#     finetuner = FunASRFinetuner(
#         train_data="/mnt/disk/wjh23/EaseDineDatasets/train_audio/train_audio_batch_2",
#         valid_data=None,
#         pretrained_name="/mnt/disk/wjh23/models/FunASR_model/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
#         output_dir="/mnt/disk/wjh23/models/FunASR_model/finetuner_models"
#     )
#     finetuner.run()

if __name__=="__main__":
    train_path = "/mnt/disk/wjh23/EaseDineDatasets/train_audio/train_audio_batch_2"
    FT = FunASRFineTuner(train_path)
    # FT.train()
    # save_path = "/mnt/disk/wjh23/models/FunASR_model/finetuner_models"
    # FT.save_model(save_path)