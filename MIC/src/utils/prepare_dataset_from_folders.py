#!/usr/bin/env python3
"""
从文件夹结构自动生成训练数据索引

数据组织格式:
/datasets/
├── 类别1/
│   ├── 样本1/
│   │   ├── *.JPG (照片)
│   │   ├── *.txt (病历)
│   │   └── processed/*.svs (病理切片)
│   └── 样本2/
└── 类别2/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random


def scan_dataset_directory(
    root_dir: str,
    processed_subdir: str = "processed",
    photo_extensions: List[str] = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
    text_extensions: List[str] = [".txt"],
    pathology_extensions: List[str] = [".svs", ".tif", ".tiff", ".kfb"],
    verbose: bool = True
) -> List[Dict]:
    """
    扫描数据集目录，生成样本列表
    
    Args:
        root_dir: 数据集根目录
        processed_subdir: processed子目录名
        photo_extensions: 照片文件扩展名列表
        text_extensions: 文本文件扩展名列表
        pathology_extensions: 病理切片文件扩展名列表
        verbose: 是否打印详细信息
        
    Returns:
        样本列表，每个样本包含 id, label, text_path, photo_paths, pathology_paths
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise ValueError(f"数据集根目录不存在: {root_dir}")
    
    samples = []
    class_stats = defaultdict(int)
    
    if verbose:
        print(f"\n🔍 扫描数据集: {root_dir}")
        print("=" * 80)
    
    # 遍历类别文件夹（第一层）
    class_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    class_dirs = sorted(class_dirs)
    
    if verbose:
        print(f"\n📁 发现 {len(class_dirs)} 个类别文件夹:")
        for i, class_dir in enumerate(class_dirs, 1):
            print(f"  {i}. {class_dir.name}")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        if verbose:
            print(f"\n📂 处理类别: {class_name}")
            print("-" * 80)
        
        # 遍历样本文件夹（第二层）
        sample_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        sample_dirs = sorted(sample_dirs)
        
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            full_sample_id = f"{class_name}/{sample_id}"
            
            # 收集照片文件
            photo_paths = []
            for ext in photo_extensions:
                photo_files = list(sample_dir.glob(f"*{ext}"))
                photo_paths.extend([
                    str(f.relative_to(root_path)) for f in sorted(photo_files)
                ])
            
            # 收集文本文件
            text_path = None
            for ext in text_extensions:
                text_files = list(sample_dir.glob(f"*{ext}"))
                if text_files:
                    # 取第一个文本文件
                    text_path = str(text_files[0].relative_to(root_path))
                    break
            
            # 收集病理切片文件（在processed子目录中）
            pathology_paths = []
            processed_dir = sample_dir / processed_subdir
            if processed_dir.exists():
                for ext in pathology_extensions:
                    path_files = list(processed_dir.glob(f"*{ext}"))
                    pathology_paths.extend([
                        {"path": str(f.relative_to(root_path)), "roi": None}
                        for f in sorted(path_files)
                    ])
            
            # 构建样本
            sample = {
                "id": full_sample_id,
                "label": class_name,
                "text_path": text_path,
                "photo_paths": photo_paths,
                "pathology_paths": pathology_paths,
            }
            
            samples.append(sample)
            class_stats[class_name] += 1
            
            if verbose:
                print(f"  ✅ {sample_id}:")
                print(f"      照片: {len(photo_paths)} 个")
                print(f"      文本: {'有' if text_path else '无'}")
                print(f"      病理: {len(pathology_paths)} 个")
    
    # 打印统计信息
    if verbose:
        print("\n" + "=" * 80)
        print("📊 数据集统计:")
        print(f"  总样本数: {len(samples)}")
        print(f"  类别分布:")
        for class_name, count in sorted(class_stats.items()):
            print(f"    - {class_name}: {count} 个样本")
    
    return samples


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    stratify: bool = True,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], Optional[List[Dict]]]:
    """
    划分数据集为训练集、验证集和测试集
    
    Args:
        samples: 样本列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例（如果为0则不创建测试集）
        stratify: 是否按类别分层划分
        random_seed: 随机种子
        
    Returns:
        (train_samples, val_samples, test_samples)
    """
    random.seed(random_seed)
    
    # 检查比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"划分比例总和必须为1.0，当前为 {total_ratio}")
    
    if stratify:
        # 按类别分组
        class_samples = defaultdict(list)
        for sample in samples:
            class_samples[sample["label"]].append(sample)
        
        train_samples = []
        val_samples = []
        test_samples = [] if test_ratio > 0 else None
        
        # 对每个类别分别划分
        for class_name, class_sample_list in class_samples.items():
            n_samples = len(class_sample_list)
            
            # 随机打乱
            shuffled = class_sample_list.copy()
            random.shuffle(shuffled)
            
            # 计算分割点
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            # 确保至少有1个样本在训练集
            if n_train == 0 and n_samples > 0:
                n_train = 1
                n_val = max(0, n_samples - n_train - (1 if test_ratio > 0 else 0))
            
            # 划分
            train_samples.extend(shuffled[:n_train])
            val_samples.extend(shuffled[n_train:n_train + n_val])
            if test_ratio > 0:
                test_samples.extend(shuffled[n_train + n_val:])
    else:
        # 不分层，直接划分
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        n_samples = len(samples)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_samples = shuffled[:n_train]
        val_samples = shuffled[n_train:n_train + n_val]
        test_samples = shuffled[n_train + n_val:] if test_ratio > 0 else None
    
    return train_samples, val_samples, test_samples


def save_index_files(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: Optional[List[Dict]],
    output_dir: str,
    verbose: bool = True
):
    """
    保存索引文件
    
    Args:
        train_samples: 训练样本
        val_samples: 验证样本
        test_samples: 测试样本（可选）
        output_dir: 输出目录
        verbose: 是否打印信息
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存训练集
    train_file = output_path / "train_index.json"
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    # 保存验证集
    val_file = output_path / "val_index.json"
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    
    # 保存测试集（如果有）
    test_file = None
    if test_samples:
        test_file = output_path / "test_index.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"\n💾 索引文件已保存:")
        print(f"  训练集: {train_file} ({len(train_samples)} 个样本)")
        print(f"  验证集: {val_file} ({len(val_samples)} 个样本)")
        if test_file:
            print(f"  测试集: {test_file} ({len(test_samples)} 个样本)")
    
    return train_file, val_file, test_file


def create_label_mapping(samples: List[Dict], output_dir: str):
    """
    创建标签映射文件
    
    Args:
        samples: 样本列表
        output_dir: 输出目录
    """
    # 收集所有类别
    class_names = sorted(set(s["label"] for s in samples))
    
    # 创建映射
    label_map = {name: idx for idx, name in enumerate(class_names)}
    idx_to_label = {idx: name for name, idx in label_map.items()}
    
    # 保存映射文件
    output_path = Path(output_dir)
    mapping_file = output_path / "label_mapping.json"
    
    mapping_data = {
        "label_map": label_map,
        "idx_to_label": idx_to_label,
        "class_names": class_names,
        "num_classes": len(class_names)
    }
    
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n🏷️  标签映射文件已保存: {mapping_file}")
    print(f"  类别数: {len(class_names)}")
    for name, idx in label_map.items():
        print(f"    {idx}: {name}")
    
    return mapping_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从文件夹结构自动生成训练数据索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 扫描数据集并生成索引文件
  python prepare_dataset_from_folders.py /path/to/datasets --output /path/to/output
  
  # 自定义划分比例
  python prepare_dataset_from_folders.py /path/to/datasets --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05
  
  # 不创建测试集
  python prepare_dataset_from_folders.py /path/to/datasets --train-ratio 0.8 --val-ratio 0.2 --test-ratio 0
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
        help="输出目录（默认为数据集根目录）"
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
        help="测试集比例（默认0.1，设为0则不创建测试集）"
    )
    
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="不使用分层划分"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="随机种子（默认42）"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，减少输出"
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = args.output if args.output else args.data_root
    
    try:
        # 1. 扫描数据集
        samples = scan_dataset_directory(
            root_dir=args.data_root,
            verbose=not args.quiet
        )
        
        if len(samples) == 0:
            print("❌ 未找到任何样本！请检查数据目录结构。")
            return
        
        # 2. 划分数据集
        train_samples, val_samples, test_samples = split_dataset(
            samples=samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify=not args.no_stratify,
            random_seed=args.random_seed
        )
        
        # 3. 保存索引文件
        save_index_files(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            output_dir=output_dir,
            verbose=not args.quiet
        )
        
        # 4. 创建标签映射
        create_label_mapping(samples, output_dir)
        
        print("\n✅ 数据准备完成！")
        print(f"\n下一步:")
        print(f"  1. 检查生成的索引文件: {output_dir}")
        print(f"  2. 运行训练: python train_from_dataset.py {args.data_root}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

