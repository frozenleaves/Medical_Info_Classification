"""
多模态医学图像分类主训练脚本
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

from .config.model_config import ModelConfig
from .config.training_config import TrainingConfig
from .data.dataloader import create_dataloaders
from .training.trainer import MultiModalTrainer
from .utils.model_utils import setup_logging, set_random_seed
from .utils.data_utils import prepare_data_splits


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多模态医学图像分类训练")

    # 基本配置
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录路径")
    parser.add_argument(
        "--output_dir", type=str, default="./experiments", help="实验输出目录"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="multimodal_exp", help="实验名称"
    )

    # 模型配置
    parser.add_argument("--model_config", type=str, help="模型配置文件路径（JSON格式）")
    parser.add_argument(
        "--training_config", type=str, help="训练配置文件路径（JSON格式）"
    )

    # 训练参数覆盖
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--num_classes", type=int, help="分类数目")

    # 设备和性能
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练")

    # 模式选择
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "resume"],
        help="运行模式",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="检查点路径（用于resume或evaluate模式）"
    )

    # 数据相关
    parser.add_argument("--train_split", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val_split", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test_split", type=float, default=0.1, help="测试集比例")

    # 其他选项
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--use_wandb", action="store_true", help="使用Weights & Biases记录"
    )
    parser.add_argument("--debug", action="store_true", help="调试模式（使用少量数据）")

    return parser.parse_args()


def load_config_file(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}: {e}")
        return {}


def get_model_config(args) -> Dict[str, Any]:
    """获取模型配置"""
    # 从默认配置开始
    config = {
        "num_classes": 6,
        "use_auxiliary_loss": True,
        "modal_dropout_prob": 0.1,
        "text_encoder": {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dim": 768,
            "max_length": 512,
            "freeze_encoder": False,
        },
        "photo_encoder": {
            "backbone": "vit_base_patch16_224",
            "pretrained": True,
            "feature_dim": 768,
            "num_heads": 8,
            "dropout": 0.1,
            "max_images": 10,
        },
        "pathology_encoder": {
            "patch_size": 256,
            "overlap": 0.1,
            "patch_backbone": "vit_large_patch16_224",
            "patch_feature_dim": 1024,
            "mil_feature_dim": 512,
            "attention_heads": 4,
            "dropout": 0.1,
            "max_patches": 10000,
        },
        "fusion": {
            "fusion_dim": 512,
            "attention_heads": 8,
            "dropout": 0.2,
            "num_layers": 2,
            "fusion_strategy": "transformer",
        },
        "classifier": {
            "type": "mlp",
            "hidden_dims": [256, 128],
            "dropout": 0.3,
            "activation": "relu",
        },
    }

    # 从配置文件更新
    if args.model_config:
        file_config = load_config_file(args.model_config)
        config.update(file_config)

    # 命令行参数覆盖
    if args.num_classes:
        config["num_classes"] = args.num_classes

    return config


def get_training_config(args) -> Dict[str, Any]:
    """获取训练配置"""
    # 从默认配置开始
    config = {
        "epochs": 100,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "scheduler": "cosine",
        "gradient_clip": 1.0,
        "accumulation_steps": 4,
        "optimizer": "adamw",
        "use_amp": True,
        "loss": {
            "type": "multimodal",
            "main_loss_type": "focal",
            "aux_loss_type": "cross_entropy",
            "focal_gamma": 2.0,
            "loss_weights": {
                "main_loss": 1.0,
                "text_aux": 0.1,
                "photo_aux": 0.1,
                "pathology_aux": 0.1,
            },
            "use_contrastive_loss": True,
            "contrastive_weight": 0.1,
        },
        "early_stopping": {"patience": 15, "min_delta": 0.001, "monitor": "val_f1"},
        "model_save": {"save_best": True, "save_last": True, "save_every_n_epochs": 10},
        "data": {
            "batch_size": 4,
            "num_workers": 4,
            "pin_memory": True,
            "train_split": 0.7,
            "val_split": 0.2,
            "test_split": 0.1,
        },
    }

    # 从配置文件更新
    if args.training_config:
        file_config = load_config_file(args.training_config)
        config.update(file_config)

    # 命令行参数覆盖
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.num_workers:
        config["data"]["num_workers"] = args.num_workers
    if args.use_amp:
        config["use_amp"] = True

    # 数据分割比例
    config["data"].update(
        {
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
        }
    )

    return config


def setup_experiment_dir(output_dir: str, experiment_name: str) -> Path:
    """设置实验目录"""
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "configs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)

    return exp_dir


def save_configs(exp_dir: Path, model_config: Dict, training_config: Dict):
    """保存配置文件"""
    # 保存模型配置
    with open(exp_dir / "configs" / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    # 保存训练配置
    with open(exp_dir / "configs" / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    args = parse_arguments()

    # 设置随机种子
    set_random_seed(args.seed)

    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"使用设备: {device}")

    # 设置实验目录
    exp_dir = setup_experiment_dir(args.output_dir, args.experiment_name)
    print(f"实验目录: {exp_dir}")

    # 获取配置
    model_config = get_model_config(args)
    training_config = get_training_config(args)

    # 更新保存目录
    training_config["model_save"]["save_dir"] = str(exp_dir / "checkpoints")

    # 保存配置
    save_configs(exp_dir, model_config, training_config)

    # 设置日志
    log_file = exp_dir / "logs" / "experiment.log"
    setup_logging(log_file)

    print("=" * 50)
    print("多模态医学图像分类训练")
    print("=" * 50)
    print(f"实验名称: {args.experiment_name}")
    print(f"数据目录: {args.data_dir}")
    print(f"模型类别数: {model_config['num_classes']}")
    print(f"训练轮数: {training_config['epochs']}")
    print(f"批次大小: {training_config['data']['batch_size']}")
    print(f"学习率: {training_config['learning_rate']}")
    print("=" * 50)

    try:
        if args.mode == "train":
            # 训练模式
            print("开始训练...")

            # 准备数据分割（如果需要）
            if not Path(args.data_dir).exists():
                raise ValueError(f"数据目录不存在: {args.data_dir}")

            # 创建数据加载器
            dataloader_config = training_config["data"].copy()
            dataloader_config.update(
                {
                    "transforms": {
                        "text": {"enable_augment": True},
                        "photo": {"enable_augment": True},
                        "pathology": {"enable_augment": True},
                    },
                    "collator": {
                        "max_photos": model_config["photo_encoder"]["max_images"],
                        "max_patches": model_config["pathology_encoder"]["max_patches"],
                    },
                    "dataset": {
                        "patch_size": model_config["pathology_encoder"]["patch_size"],
                        "overlap": model_config["pathology_encoder"]["overlap"],
                        "max_patches": model_config["pathology_encoder"]["max_patches"],
                    },
                }
            )

            # 调试模式：使用少量数据
            if args.debug:
                print("调试模式: 使用少量数据")
                dataloader_config["batch_size"] = 2
                training_config["epochs"] = 3
                training_config["early_stopping"]["patience"] = 2

            dataloaders = create_dataloaders(args.data_dir, dataloader_config)

            if not dataloaders:
                raise ValueError("无法创建数据加载器，请检查数据格式")

            # 创建训练器
            trainer = MultiModalTrainer(
                model_config=model_config,
                training_config=training_config,
                device=device,
                use_wandb=args.use_wandb,
            )

            # 开始训练
            train_dataloader = dataloaders.get("train")
            val_dataloader = dataloaders.get("val")

            if not train_dataloader:
                raise ValueError("训练数据加载器为空")

            if not val_dataloader:
                print("警告: 验证数据加载器为空，使用训练集进行验证")
                val_dataloader = train_dataloader

            training_history = trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                resume_checkpoint=args.checkpoint,
            )

            # 保存训练历史
            history_file = exp_dir / "results" / "training_history.json"
            with open(history_file, "w", encoding="utf-8") as f:
                # 处理numpy类型
                serializable_history = {}
                for key, value in training_history.items():
                    if isinstance(value, list):
                        serializable_history[key] = [
                            (
                                {
                                    k: (
                                        float(v)
                                        if isinstance(v, (np.floating, np.integer))
                                        else v
                                    )
                                    for k, v in item.items()
                                }
                                if isinstance(item, dict)
                                else item
                            )
                            for item in value
                        ]
                    else:
                        serializable_history[key] = value

                json.dump(serializable_history, f, ensure_ascii=False, indent=2)

            print(f"训练历史已保存到: {history_file}")

            # 测试集评估
            test_dataloader = dataloaders.get("test")
            if test_dataloader and len(test_dataloader) > 0:
                print("开始测试集评估...")
                test_results = trainer.evaluate(test_dataloader)

                # 保存测试结果
                test_file = exp_dir / "results" / "test_results.json"
                with open(test_file, "w", encoding="utf-8") as f:
                    # 处理numpy类型
                    serializable_results = {}
                    for key, value in test_results.items():
                        if isinstance(value, np.ndarray):
                            serializable_results[key] = value.tolist()
                        elif isinstance(value, dict):
                            serializable_results[key] = {
                                k: (
                                    float(v)
                                    if isinstance(v, (np.floating, np.integer))
                                    else v
                                )
                                for k, v in value.items()
                            }
                        else:
                            serializable_results[key] = value

                    json.dump(serializable_results, f, ensure_ascii=False, indent=2)

                print(f"测试结果已保存到: {test_file}")

                # 打印主要指标
                main_metrics = test_results.get("metrics", {}).get("main", {})
                print(f"\n测试集结果:")
                print(f"  准确率: {main_metrics.get('accuracy', 'N/A'):.4f}")
                print(f"  F1分数: {main_metrics.get('f1_score', 'N/A'):.4f}")
                print(f"  精确率: {main_metrics.get('precision', 'N/A'):.4f}")
                print(f"  召回率: {main_metrics.get('recall', 'N/A'):.4f}")

        elif args.mode == "evaluate":
            # 评估模式
            if not args.checkpoint:
                raise ValueError("评估模式需要提供checkpoint参数")

            print(f"开始评估模型: {args.checkpoint}")

            # 创建数据加载器
            dataloader_config = training_config["data"].copy()
            dataloaders = create_dataloaders(args.data_dir, dataloader_config)

            # 创建训练器并加载模型
            trainer = MultiModalTrainer(
                model_config=model_config,
                training_config=training_config,
                device=device,
                use_wandb=False,
            )

            trainer._load_checkpoint(args.checkpoint)

            # 评估
            test_dataloader = dataloaders.get("test") or dataloaders.get("val")
            if test_dataloader:
                results = trainer.evaluate(test_dataloader)

                # 打印结果
                main_metrics = results.get("metrics", {}).get("main", {})
                print(f"评估结果:")
                print(f"  准确率: {main_metrics.get('accuracy', 'N/A'):.4f}")
                print(f"  F1分数: {main_metrics.get('f1_score', 'N/A'):.4f}")
                print(f"  精确率: {main_metrics.get('precision', 'N/A'):.4f}")
                print(f"  召回率: {main_metrics.get('recall', 'N/A'):.4f}")

                # 保存结果
                eval_file = exp_dir / "results" / "evaluation_results.json"
                with open(eval_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

                print(f"评估结果已保存到: {eval_file}")
            else:
                print("错误: 无法找到测试数据")

        elif args.mode == "resume":
            # 恢复训练模式
            if not args.checkpoint:
                raise ValueError("恢复训练模式需要提供checkpoint参数")

            print(f"恢复训练: {args.checkpoint}")

            # 创建数据加载器和训练器（与训练模式相同）
            dataloader_config = training_config["data"].copy()
            dataloaders = create_dataloaders(args.data_dir, dataloader_config)

            trainer = MultiModalTrainer(
                model_config=model_config,
                training_config=training_config,
                device=device,
                use_wandb=args.use_wandb,
            )

            # 恢复训练
            training_history = trainer.train(
                train_dataloader=dataloaders["train"],
                val_dataloader=dataloaders.get("val", dataloaders["train"]),
                resume_checkpoint=args.checkpoint,
            )

            print("恢复训练完成!")

        print("实验完成!")

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"实验失败: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
