#!/usr/bin/env python3
"""
从数据集根目录开始训练

自动扫描数据集、生成索引、开始训练的一站式脚本
"""

import sys
import json
import argparse
from pathlib import Path
import torch

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from MIC.src.utils.prepare_dataset_from_folders import (
    scan_dataset_directory,
    split_dataset,
    save_index_files,
    create_label_mapping
)
from MIC.src.data.dataset import MultiModalMedicalDataset
from MIC.src.data.dataloader import create_dataloader
from MIC.src.data.transforms import get_transforms
from MIC.src.models.multimodal_model import MultiModalMedicalClassifier
from MIC.src.training.trainer import MultiModalTrainer
from MIC.src.training.loss import create_loss_function


def prepare_data(
    data_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    force_rebuild: bool = False,
    random_seed: int = 42
):
    """
    准备数据集索引文件
    
    Args:
        data_root: 数据根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        force_rebuild: 是否强制重新生成索引
        random_seed: 随机种子
        
    Returns:
        (train_index_file, val_index_file, label_mapping_file)
    """
    data_path = Path(data_root)
    
    # 检查是否已存在索引文件
    train_index = data_path / "train_index.json"
    val_index = data_path / "val_index.json"
    label_mapping = data_path / "label_mapping.json"
    
    if not force_rebuild and train_index.exists() and val_index.exists():
        print(f"\n📁 发现已存在的索引文件:")
        print(f"  - {train_index}")
        print(f"  - {val_index}")
        
        if label_mapping.exists():
            print(f"  - {label_mapping}")
        
        response = input("\n是否使用现有索引文件？(y/n，直接回车默认为y): ").strip().lower()
        if response in ["", "y", "yes"]:
            return train_index, val_index, label_mapping
    
    print("\n🔧 准备数据集索引...")
    
    # 1. 扫描数据集
    samples = scan_dataset_directory(data_root, verbose=True)
    
    if len(samples) == 0:
        raise ValueError("未找到任何样本！请检查数据目录结构。")
    
    # 2. 划分数据集
    train_samples, val_samples, test_samples = split_dataset(
        samples=samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=True,
        random_seed=random_seed
    )
    
    print(f"\n📊 数据集划分:")
    print(f"  训练集: {len(train_samples)} 个样本")
    print(f"  验证集: {len(val_samples)} 个样本")
    if test_samples:
        print(f"  测试集: {len(test_samples)} 个样本")
    
    # 3. 保存索引文件
    save_index_files(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        output_dir=data_root,
        verbose=True
    )
    
    # 4. 创建标签映射
    create_label_mapping(samples, data_root)
    
    return train_index, val_index, label_mapping


def load_or_create_config(
    data_root: str,
    num_classes: int,
    config_file: str = None
) -> dict:
    """
    加载或创建配置文件
    
    Args:
        data_root: 数据根目录
        num_classes: 类别数
        config_file: 配置文件路径（可选）
        
    Returns:
        配置字典
    """
    if config_file and Path(config_file).exists():
        print(f"\n📄 加载配置文件: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 创建默认配置
    print(f"\n⚙️  使用默认配置 (类别数: {num_classes})")
    
    config = {
        "model": {
            "num_classes": num_classes,
            "use_pretrained": True,
            "modal_dropout_prob": 0.1,
            "use_auxiliary_loss": True,
            "auxiliary_loss_weight": 0.3,
            
            "text_encoder": {
                "model_name": "hfl/chinese-roberta-wwm-ext",
                "local_model_path": None,
                "max_length": 512,
                "hidden_size": 768,
            },
            
            "photo_encoder": {
                "model_name": "google/vit-base-patch16-224",
                "local_model_path": None,
                "hidden_size": 768,
                "use_pretrained": True,
            },
            
            "pathology_encoder": {
                "model_name": "owkin/phikon",
                "local_model_path": None,
                "hidden_size": 768,
                "aggregation_method": "attention",
                "use_pretrained": True,
            },
            
            "fusion": {
                "method": "cross_attention",
                "hidden_size": 768,
                "num_heads": 8,
                "num_layers": 2,
                "dropout": 0.1,
            },
            
            "classifier": {
                "hidden_sizes": [512, 256],
                "dropout": 0.3,
                "use_batch_norm": False,
            }
        },
        
        "training": {
            "num_epochs": 50,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "gradient_clip_val": 1.0,
            "early_stopping_patience": 10,
            "save_top_k": 3,
            
            "optimizer": {
                "type": "adamw",
                "lr": 1e-4,
                "weight_decay": 0.01,
            },
            
            "scheduler": {
                "type": "cosine",
                "warmup_steps": 100,
            },
            
            "loss": {
                "type": "multimodal",
                "num_classes": num_classes,
                "class_weights": None,
                "label_smoothing": 0.1,
            }
        },
        
        "data": {
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            
            "pathology_config": {
                "patch_size": 512,
                "extract_levels": "all",
                "overlap": 0.1,
                "max_patches": 1000,
                "filter_blank_patches": True,
            },
            
            "transforms": {
                "text": {
                    "max_length": 512,
                },
                "photo": {
                    "input_size": 224,
                },
                "pathology": {
                    "input_size": 224,
                    "patch_size": 512,
                }
            }
        }
    }
    
    return config


def create_dataloaders(data_root: str, config: dict):
    """
    创建数据加载器
    
    Args:
        data_root: 数据根目录
        config: 配置字典
        
    Returns:
        (train_loader, val_loader)
    """
    print("\n📦 创建数据加载器...")
    
    data_path = Path(data_root)
    
    # 获取数据变换
    transforms = get_transforms(config.get("data", {}).get("transforms", {}))
    
    # 创建训练集
    train_dataset = MultiModalMedicalDataset(
        data_dir=data_root,
        index_file=str(data_path / "train_index.json"),
        transforms=transforms,
        config=config.get("data", {}).get("pathology_config", {})
    )
    
    # 创建验证集
    val_dataset = MultiModalMedicalDataset(
        data_dir=data_root,
        index_file=str(data_path / "val_index.json"),
        transforms=transforms,
        config=config.get("data", {}).get("pathology_config", {})
    )
    
    print(f"  ✅ 训练集: {len(train_dataset)} 个样本")
    print(f"  ✅ 验证集: {len(val_dataset)} 个样本")
    
    # 创建数据加载器
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config.get("data", {}).get("num_workers", 4),
        pin_memory=config.get("data", {}).get("pin_memory", True),
    )
    
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config.get("data", {}).get("num_workers", 4),
        pin_memory=config.get("data", {}).get("pin_memory", True),
    )
    
    return train_loader, val_loader, train_dataset


def train(
    data_root: str,
    output_dir: str = None,
    config_file: str = None,
    resume_from: str = None,
    **kwargs
):
    """
    训练模型
    
    Args:
        data_root: 数据根目录
        output_dir: 输出目录
        config_file: 配置文件路径
        resume_from: 恢复训练的checkpoint路径
        **kwargs: 其他参数（覆盖配置）
    """
    print("\n" + "=" * 80)
    print("🚀 开始训练多模态医学图像分类模型")
    print("=" * 80)
    
    # 1. 准备数据
    train_index, val_index, label_mapping = prepare_data(
        data_root=data_root,
        train_ratio=kwargs.get("train_ratio", 0.7),
        val_ratio=kwargs.get("val_ratio", 0.2),
        test_ratio=kwargs.get("test_ratio", 0.1),
        force_rebuild=kwargs.get("force_rebuild", False),
        random_seed=kwargs.get("random_seed", 42)
    )
    
    # 2. 加载标签映射
    with open(label_mapping, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    num_classes = label_data["num_classes"]
    print(f"\n🏷️  类别数: {num_classes}")
    for name, idx in label_data["label_map"].items():
        print(f"  {idx}: {name}")
    
    # 3. 加载配置
    config = load_or_create_config(data_root, num_classes, config_file)
    
    # 覆盖配置参数
    if "batch_size" in kwargs:
        config["training"]["batch_size"] = kwargs["batch_size"]
    if "learning_rate" in kwargs:
        config["training"]["learning_rate"] = kwargs["learning_rate"]
    if "num_epochs" in kwargs:
        config["training"]["num_epochs"] = kwargs["num_epochs"]
    
    # 4. 创建数据加载器
    train_loader, val_loader, train_dataset = create_dataloaders(data_root, config)
    
    # 5. 创建模型
    print("\n🔨 创建模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    
    model = MultiModalMedicalClassifier(config["model"])
    model = model.to(device)
    
    print(f"  ✅ 模型已创建")
    
    # 6. 创建损失函数
    loss_fn = create_loss_function(config["training"]["loss"])
    
    # 7. 设置输出目录
    if output_dir is None:
        output_dir = str(Path(data_root) / "outputs")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n💾 输出目录: {output_dir}")
    
    # 8. 创建训练器
    print("\n🎯 初始化训练器...")
    trainer = MultiModalTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        config=config["training"],
        device=device,
        output_dir=output_dir,
    )
    
    # 9. 开始训练
    print("\n" + "=" * 80)
    print("🏃 开始训练...")
    print("=" * 80)
    
    try:
        trainer.train(
            num_epochs=config["training"]["num_epochs"],
            resume_from=resume_from
        )
        
        print("\n" + "=" * 80)
        print("✅ 训练完成！")
        print("=" * 80)
        print(f"\n模型已保存到: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print(f"最新的checkpoint已保存到: {output_dir}")
    
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从数据集根目录开始训练多模态医学图像分类模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置训练
  python train_from_dataset.py /path/to/datasets
  
  # 指定输出目录
  python train_from_dataset.py /path/to/datasets --output ./outputs
  
  # 使用自定义配置
  python train_from_dataset.py /path/to/datasets --config config.json
  
  # 恢复训练
  python train_from_dataset.py /path/to/datasets --resume ./outputs/checkpoint_epoch_10.pth
  
  # 调整超参数
  python train_from_dataset.py /path/to/datasets --batch-size 8 --learning-rate 2e-4 --epochs 100
        """
    )
    
    parser.add_argument(
        "data_root",
        type=str,
        help="数据集根目录路径"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（默认为 data_root/outputs）"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（JSON格式）"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的checkpoint路径"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批次大小"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="学习率"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="训练集比例（默认0.7）"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例（默认0.2）"
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="测试集比例（默认0.1）"
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="强制重新生成索引文件"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="随机种子（默认42）"
    )
    
    args = parser.parse_args()
    
    # 构建kwargs
    kwargs = {
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "force_rebuild": args.force_rebuild,
        "random_seed": args.random_seed,
    }
    
    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    if args.learning_rate:
        kwargs["learning_rate"] = args.learning_rate
    if args.epochs:
        kwargs["num_epochs"] = args.epochs
    
    # 开始训练
    train(
        data_root=args.data_root,
        output_dir=args.output,
        config_file=args.config,
        resume_from=args.resume,
        **kwargs
    )


if __name__ == "__main__":
    main()

