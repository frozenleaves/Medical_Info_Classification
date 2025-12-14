"""
训练器模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from tqdm import tqdm
import wandb

from .loss import create_loss_function
from .metrics import MultiModalMetrics
from ..models.multimodal_model import create_multimodal_model


class MultiModalTrainer:
    """
    多模态医学信息分类训练器
    """

    def __init__(
        self,
        model_config: Dict,
        training_config: Dict,
        device: str = "cuda",
        use_wandb: bool = True,
    ):

        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        self.use_wandb = use_wandb

        # 基本训练配置
        self.num_epochs = training_config.get("epochs", 100)
        self.learning_rate = training_config.get("learning_rate", 2e-4)
        self.weight_decay = training_config.get("weight_decay", 1e-4)
        self.gradient_clip = training_config.get("gradient_clip", 1.0)
        self.accumulation_steps = training_config.get("accumulation_steps", 4)

        # 早停配置
        self.early_stopping = training_config.get("early_stopping", {})
        self.patience = self.early_stopping.get("patience", 15)
        self.min_delta = self.early_stopping.get("min_delta", 0.001)
        self.monitor_metric = self.early_stopping.get("monitor", "val_f1")

        # 模型保存配置
        self.save_config = training_config.get("model_save", {})
        self.save_dir = Path(self.save_config.get("save_dir", "./checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 创建模型
        self.model = create_multimodal_model(model_config)
        self.model.to(device)

        # 创建损失函数
        loss_config = training_config.get("loss", {})
        loss_config["num_classes"] = model_config.get("num_classes", 6)
        self.criterion = create_loss_function(loss_config)

        # 创建优化器
        self.optimizer = self._create_optimizer()

        # 创建学习率调度器
        self.scheduler = self._create_scheduler()

        # 混合精度训练
        self.use_amp = training_config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None

        # 评估指标
        class_names = model_config.get(
            "class_names",
            [f"Class_{i}" for i in range(model_config.get("num_classes", 6))],
        )
        self.metrics = MultiModalMetrics(
            model_config.get("num_classes", 6), class_names
        )

        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }

        # 日志设置
        self._setup_logging()

        # 初始化wandb（如果启用）
        if self.use_wandb:
            self._init_wandb()

        print(f"训练器初始化完成:")
        print(f"  模型参数数量: {self._count_parameters()}")
        print(f"  设备: {device}")
        print(f"  混合精度: {self.use_amp}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  批次累积: {self.accumulation_steps}")

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_type = self.training_config.get("optimizer", "adamw")

        if optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

    def _create_scheduler(self) -> Optional[object]:
        """创建学习率调度器"""
        scheduler_type = self.training_config.get("scheduler", "cosine")

        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs, eta_min=self.learning_rate * 0.01
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_config.get("step_size", 30),
                gamma=self.training_config.get("gamma", 0.1),
            )
        elif scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max" if "f1" in self.monitor_metric else "min",
                factor=0.5,
                patience=10,
                verbose=True,
            )
        elif scheduler_type == "warmup_cosine":
            return self._create_warmup_scheduler()
        else:
            return None

    def _create_warmup_scheduler(self):
        """创建带warmup的余弦调度器"""
        from torch.optim.lr_scheduler import LambdaLR

        warmup_epochs = self.training_config.get("warmup_epochs", 5)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (self.num_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def _setup_logging(self):
        """设置日志"""
        log_dir = self.save_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _init_wandb(self):
        """初始化Weights & Biases"""
        try:
            wandb.init(
                project="multimodal-medical-classification",
                config={**self.model_config, **self.training_config},
                name=f"multimodal_run_{int(time.time())}",
            )
            wandb.watch(self.model, log="all", log_freq=100)
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False

    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        resume_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            resume_checkpoint: 恢复训练的检查点路径

        Returns:
            训练历史字典
        """
        # 从数据集获取真实的标签映射
        if hasattr(train_dataloader.dataset, "label_map"):
            self.label_map = train_dataloader.dataset.label_map
            # 创建反向映射：索引 -> 标签名
            self.idx_to_label = {idx: label for label, idx in self.label_map.items()}

            # 获取实际类别数
            actual_num_classes = len(self.label_map)
            model_num_classes = self.model_config.get("num_classes", 6)

            # 检查类别数是否匹配
            if actual_num_classes != model_num_classes:
                self.logger.warning(
                    f"⚠️  数据集类别数({actual_num_classes})与模型定义({model_num_classes})不一致！"
                )

                if actual_num_classes > model_num_classes:
                    # 数据类别多于模型类别 - 这是严重错误
                    raise ValueError(
                        f"数据集有{actual_num_classes}个类别，但模型只定义了{model_num_classes}个类别！"
                        f"请修改model_config中的num_classes为{actual_num_classes}或更大。"
                    )
                else:
                    # 数据类别少于模型类别 - 警告但可以继续
                    self.logger.warning(
                        f"数据集只有{actual_num_classes}个类别，但模型定义了{model_num_classes}个类别。"
                    )
                    self.logger.warning(
                        f"建议将model_config中的num_classes改为{actual_num_classes}以避免浪费参数。"
                    )
                    self.logger.warning(
                        f"当前将使用数据集的{actual_num_classes}个类别进行训练。"
                    )

            # 构建class_names
            # 使用模型定义的类别数，为缺失的类别填充占位符
            self.class_names = []
            for i in range(model_num_classes):
                if i in self.idx_to_label:
                    self.class_names.append(self.idx_to_label[i])
                else:
                    # 未使用的类别（数据中没有）
                    placeholder = f"Unused_Class_{i}"
                    self.class_names.append(placeholder)
                    self.logger.warning(
                        f"类别索引{i}在数据中不存在，使用占位符: {placeholder}"
                    )

            self.logger.info(f"从数据集获取标签映射: {self.label_map}")
            self.logger.info(
                f"类别名称列表({len(self.class_names)}): {self.class_names}"
            )

        else:
            self.logger.warning("数据集没有label_map属性，使用默认类别名称")
            self.label_map = None
            self.class_names = [
                f"Class_{i}" for i in range(self.model_config.get("num_classes", 6))
            ]

        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)

        self.logger.info("开始训练...")

        # 训练循环
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_results = self._train_epoch(train_dataloader)

            # 验证
            val_results = self._validate_epoch(val_dataloader)

            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results["metrics"]["main"]["f1_score"])
                else:
                    self.scheduler.step()

            # 记录历史
            self.training_history["train_loss"].append(train_results["loss"])
            self.training_history["val_loss"].append(val_results["loss"])
            self.training_history["train_metrics"].append(train_results["metrics"])
            self.training_history["val_metrics"].append(val_results["metrics"])

            # 日志记录
            self._log_epoch_results(epoch, train_results, val_results)

            # 模型保存
            is_best = self._save_checkpoint(val_results["metrics"]["main"])

            # 早停检查
            if self._check_early_stopping(val_results["metrics"]["main"]):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # 加载最佳模型
        best_checkpoint = self.save_dir / "best_model.pth"
        if best_checkpoint.exists():
            self._load_checkpoint(str(best_checkpoint))

        self.logger.info("训练完成!")

        return self.training_history

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """训练一个epoch"""
        self.model.train()
        self.metrics.reset()

        epoch_loss = 0.0
        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 移动数据到设备
                batch = self._move_batch_to_device(batch)
                targets = batch["labels"]

                # 前向传播
                with autocast(enabled=self.use_amp):
                    outputs = self.model(batch)
                    loss_dict = self.criterion(
                        outputs, targets, batch.get("modal_availability")
                    )
                    loss = loss_dict["total_loss"] / self.accumulation_steps

                # 反向传播
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 梯度累积
                if (
                    batch_idx + 1
                ) % self.accumulation_steps == 0 or batch_idx == num_batches - 1:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                # 更新指标
                epoch_loss += loss.item() * self.accumulation_steps

                # 转换为numpy用于指标计算
                outputs_np = self._convert_outputs_to_numpy(outputs)
                targets_np = targets.cpu().numpy()
                # modal_availability应该是[batch_size, 3]，3代表3个模态
                modal_availability_np = batch.get(
                    "modal_availability",
                    torch.ones(len(targets), 3, device=targets.device),
                )
                modal_availability_np = modal_availability_np.cpu().numpy()

                self.metrics.update(outputs_np, targets_np, modal_availability_np)

                # 更新进度条
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item() * self.accumulation_steps:.4f}",
                        "lr": f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    }
                )

            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {e}")
                continue

        # 计算epoch指标
        epoch_metrics = self.metrics.compute()
        avg_loss = epoch_loss / num_batches

        return {"loss": avg_loss, "metrics": epoch_metrics}

    def _validate_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """验证一个epoch"""
        self.model.eval()
        self.metrics.reset()

        epoch_loss = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
                try:
                    # 移动数据到设备
                    batch = self._move_batch_to_device(batch)
                    targets = batch["labels"]

                    # 前向传播
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(batch)
                        loss_dict = self.criterion(
                            outputs, targets, batch.get("modal_availability")
                        )
                        loss = loss_dict["total_loss"]

                    # 更新损失
                    epoch_loss += loss.item()

                    # 更新指标
                    outputs_np = self._convert_outputs_to_numpy(outputs)
                    targets_np = targets.cpu().numpy()
                    # modal_availability应该是[batch_size, 3]，3代表3个模态
                    modal_availability_np = batch.get(
                        "modal_availability",
                        torch.ones(len(targets), 3, device=targets.device),
                    )
                    modal_availability_np = modal_availability_np.cpu().numpy()

                    self.metrics.update(outputs_np, targets_np, modal_availability_np)

                except Exception as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue

        # 计算epoch指标
        epoch_metrics = self.metrics.compute()
        avg_loss = epoch_loss / num_batches

        return {"loss": avg_loss, "metrics": epoch_metrics}

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """将批次数据移动到设备"""
        moved_batch = {}

        for key, value in batch.items():
            if key == "text":
                moved_batch[key] = value  # 文本保持原样，在encoder内部处理
            elif isinstance(value, dict):
                moved_batch[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        moved_batch[key][sub_key] = sub_value.to(self.device)
                    else:
                        moved_batch[key][sub_key] = sub_value
            elif isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value

        return moved_batch

    def _convert_outputs_to_numpy(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """将模型输出转换为numpy格式"""
        converted = {}

        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                converted[key] = value.detach().cpu().numpy()
            elif isinstance(value, dict):
                converted[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        converted[key][sub_key] = sub_value.detach().cpu().numpy()
                    else:
                        converted[key][sub_key] = sub_value
            else:
                converted[key] = value

        return converted

    def _log_epoch_results(self, epoch: int, train_results: Dict, val_results: Dict):
        """记录epoch结果"""
        train_loss = train_results["loss"]
        val_loss = val_results["loss"]

        train_f1 = train_results["metrics"]["main"].get("f1_score", 0.0)
        val_f1 = val_results["metrics"]["main"].get("f1_score", 0.0)

        # 控制台输出
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        )

        # Wandb记录
        if self.use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/f1_score": train_f1,
                "val/loss": val_loss,
                "val/f1_score": val_f1,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            # 添加详细指标
            for metric_name, value in train_results["metrics"]["main"].items():
                if isinstance(value, (int, float)):
                    log_dict[f"train/{metric_name}"] = value

            for metric_name, value in val_results["metrics"]["main"].items():
                if isinstance(value, (int, float)):
                    log_dict[f"val/{metric_name}"] = value

            wandb.log(log_dict)

    def _save_checkpoint(self, val_metrics: Dict[str, float]) -> bool:
        """保存检查点"""
        current_metric = val_metrics.get("f1_score", 0.0)
        is_best = current_metric > self.best_metric

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_metric": self.best_metric,
            "current_metric": current_metric,
            "model_config": self.model_config,
            "training_config": self.training_config,
            # 保存标签映射信息
            "label_map": getattr(
                self, "label_map", None
            ),  # {'class_A': 0, 'class_B': 1, ...}
            "class_names": getattr(
                self, "class_names", None
            ),  # ['class_A', 'class_B', ...]
            "idx_to_label": getattr(
                self, "idx_to_label", None
            ),  # {0: 'class_A', 1: 'class_B', ...}
        }

        # 保存最新检查点
        torch.save(checkpoint, self.save_dir / "latest_model.pth")

        # 保存最佳检查点
        if is_best:
            self.best_metric = current_metric
            torch.save(checkpoint, self.save_dir / "best_model.pth")
            self.logger.info(
                f"New best model saved with {self.monitor_metric}: {current_metric:.4f}"
            )

        # 定期保存
        if self.save_config.get("save_every_n_epochs", 0) > 0:
            if self.current_epoch % self.save_config["save_every_n_epochs"] == 0:
                torch.save(
                    checkpoint, self.save_dir / f"epoch_{self.current_epoch}.pth"
                )

        return is_best

    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint["best_metric"]

        self.logger.info(
            f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.current_epoch}"
        )

    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """检查是否需要早停"""
        current_metric = val_metrics.get("f1_score", 0.0)

        if current_metric > self.best_metric + self.min_delta:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            return True

        return False

    def evaluate(self, test_dataloader: DataLoader) -> Dict[str, Any]:
        """在测试集上评估模型"""
        self.logger.info("开始测试集评估...")

        self.model.eval()
        self.metrics.reset()

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):
                batch = self._move_batch_to_device(batch)
                targets = batch["labels"]

                outputs = self.model(batch)

                # 收集预测结果
                predictions = outputs["predictions"].argmax(dim=1).cpu().numpy()
                probabilities = outputs["predictions"].cpu().numpy()
                targets_np = targets.cpu().numpy()

                all_predictions.extend(predictions)
                all_targets.extend(targets_np)
                all_probabilities.extend(probabilities)

                # 更新指标
                outputs_np = self._convert_outputs_to_numpy(outputs)
                # modal_availability应该是[batch_size, 3]，3代表3个模态
                modal_availability_np = batch.get(
                    "modal_availability",
                    torch.ones(len(targets), 3, device=targets.device),
                )
                modal_availability_np = modal_availability_np.cpu().numpy()

                self.metrics.update(outputs_np, targets_np, modal_availability_np)

        # 计算最终指标
        test_metrics = self.metrics.compute()

        # 生成详细报告
        test_report = self.metrics.main_metrics.get_classification_report()
        confusion_matrix = self.metrics.main_metrics.get_confusion_matrix()

        results = {
            "metrics": test_metrics,
            "classification_report": test_report,
            "confusion_matrix": confusion_matrix.tolist(),
            "predictions": all_predictions,
            "targets": all_targets,
            "probabilities": all_probabilities,
        }

        self.logger.info("测试集评估完成!")
        self.logger.info(f"测试集F1分数: {test_metrics['main'].get('f1_score', 'N/A')}")

        return results


if __name__ == "__main__":
    # 测试训练器
    from config.model_config import ModelConfig
    from config.training_config import TrainingConfig

    # 简化配置用于测试
    model_config = {
        "num_classes": 6,
        "text_encoder": {"model_name": "bert-base-chinese", "embedding_dim": 768},
        "photo_encoder": {"backbone": "resnet18", "feature_dim": 512},
        "pathology_encoder": {"patch_backbone": "resnet18", "mil_feature_dim": 512},
        "fusion": {"fusion_dim": 512},
        "classifier": {"type": "mlp", "hidden_dims": [256, 128]},
    }

    training_config = {
        "epochs": 2,  # 测试用小数值
        "learning_rate": 1e-3,
        "use_amp": False,  # 测试时关闭
        "loss": {"type": "multimodal", "main_loss_type": "cross_entropy"},
    }

    try:
        trainer = MultiModalTrainer(
            model_config=model_config,
            training_config=training_config,
            device="cpu",  # 测试用CPU
            use_wandb=False,  # 测试时关闭wandb
        )

        print("训练器测试成功!")
        print(f"模型参数数量: {trainer._count_parameters()}")

    except Exception as e:
        print(f"训练器测试失败: {e}")
        import traceback

        traceback.print_exc()
