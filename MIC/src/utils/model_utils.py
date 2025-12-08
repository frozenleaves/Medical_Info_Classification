"""
模型相关工具函数
"""

import torch
import numpy as np
import random
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any


def set_random_seed(seed: int = 42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置CuDNN确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"随机种子设置为: {seed}")


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO):
    """设置日志记录"""
    # 创建日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 添加文件处理器（如果指定了文件）
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        print(f"日志记录到文件: {log_file}")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
    }


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """获取模型大小信息"""
    # 计算参数大小（MB）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())

    # 计算缓冲区大小（MB）
    buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())

    total_size = param_size + buffer_size

    return {
        "parameters_mb": param_size / (1024**2),
        "buffers_mb": buffer_size / (1024**2),
        "total_mb": total_size / (1024**2),
    }


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """打印模型摘要信息"""
    print(f"\n{'='*50}")
    print(f"{model_name} 摘要")
    print(f"{'='*50}")

    # 参数统计
    param_stats = count_parameters(model)
    print(f"参数统计:")
    print(f"  总参数数: {param_stats['total']:,}")
    print(f"  可训练参数: {param_stats['trainable']:,}")
    print(f"  冻结参数: {param_stats['frozen']:,}")

    # 模型大小
    size_stats = get_model_size(model)
    print(f"\n模型大小:")
    print(f"  参数大小: {size_stats['parameters_mb']:.2f} MB")
    print(f"  缓冲区大小: {size_stats['buffers_mb']:.2f} MB")
    print(f"  总大小: {size_stats['total_mb']:.2f} MB")

    print(f"{'='*50}\n")


def freeze_model_parts(model: torch.nn.Module, freeze_config: Dict[str, bool]):
    """冻结模型的指定部分

    Args:
        model: 要冻结的模型
        freeze_config: 冻结配置，例如:
            {
                'text_encoder': True,
                'photo_encoder.backbone': True,
                'pathology_encoder': False
            }
    """
    frozen_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += 1

        # 检查是否需要冻结这个参数
        should_freeze = False
        for freeze_key, freeze_value in freeze_config.items():
            if freeze_key in name and freeze_value:
                should_freeze = True
                break

        if should_freeze:
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True

    print(f"冻结配置应用完成: {frozen_params}/{total_params} 参数被冻结")

    return frozen_params, total_params


def get_device_info():
    """获取设备信息"""
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory
            device_info[f"device_{i}"] = {
                "name": device_name,
                "memory_gb": device_memory / (1024**3),
            }

    return device_info


def print_device_info():
    """打印设备信息"""
    info = get_device_info()

    print(f"\n{'='*50}")
    print("设备信息")
    print(f"{'='*50}")
    print(f"CUDA可用: {info['cuda_available']}")

    if info["cuda_available"]:
        print(f"CUDA版本: {info['cuda_version']}")
        print(f"GPU数量: {info['device_count']}")

        for i in range(info["device_count"]):
            if f"device_{i}" in info:
                device = info[f"device_{i}"]
                print(f"  GPU {i}: {device['name']} ({device['memory_gb']:.1f} GB)")
    else:
        print("使用CPU进行训练")

    print(f"{'='*50}\n")


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    filepath: Path,
):
    """保存模型检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "model_config": model_config,
        "training_config": training_config,
    }

    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath}")


def load_model_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location=device)

    # 加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"])

    # 加载优化器状态（如果提供）
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 加载调度器状态（如果提供）
    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"]
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    print(f"检查点已加载: {filepath}")
    print(f"恢复到epoch {epoch}")

    return epoch, metrics


def convert_model_to_half(model: torch.nn.Module):
    """将模型转换为半精度"""
    model.half()
    print("模型已转换为半精度 (FP16)")
    return model


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """获取当前学习率"""
    return optimizer.param_groups[0]["lr"]


def adjust_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    """调整学习率"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"学习率调整为: {lr}")


if __name__ == "__main__":
    # 测试工具函数
    print("测试模型工具函数...")

    # 测试设备信息
    print_device_info()

    # 测试随机种子
    set_random_seed(42)

    # 创建测试模型
    test_model = torch.nn.Sequential(
        torch.nn.Linear(100, 50), torch.nn.ReLU(), torch.nn.Linear(50, 10)
    )

    # 测试模型摘要
    print_model_summary(test_model, "测试模型")

    # 测试参数冻结
    freeze_config = {"0.weight": True}  # 冻结第一层权重
    freeze_model_parts(test_model, freeze_config)

    print("模型工具函数测试完成!")
