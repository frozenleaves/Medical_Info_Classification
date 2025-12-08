"""
数据相关工具函数
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def create_data_index_from_directory(
    data_dir: str,
    output_dir: str,
    text_subdir: str = "texts",
    photo_subdir: str = "photos",
    pathology_subdir: str = "pathology",
    label_file: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """
    从目录结构创建数据索引文件

    Args:
        data_dir: 数据根目录
        output_dir: 输出目录
        text_subdir: 文本子目录名
        photo_subdir: 照片子目录名
        pathology_subdir: 病理子目录名
        label_file: 标签文件路径（CSV或JSON格式）

    Returns:
        创建的索引字典
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载标签信息
    labels_dict = {}
    if label_file and Path(label_file).exists():
        if label_file.endswith(".csv"):
            labels_df = pd.read_csv(label_file)
            if "id" in labels_df.columns and "label" in labels_df.columns:
                labels_dict = dict(zip(labels_df["id"], labels_df["label"]))
        elif label_file.endswith(".json"):
            with open(label_file, "r", encoding="utf-8") as f:
                labels_dict = json.load(f)

    # 收集所有样本
    samples = []
    text_dir = data_path / text_subdir
    photo_dir = data_path / photo_subdir
    pathology_dir = data_path / pathology_subdir

    # 从文本文件获取样本ID
    sample_ids = set()
    if text_dir.exists():
        for text_file in text_dir.glob("*.txt"):
            sample_id = text_file.stem
            sample_ids.add(sample_id)

    # 从照片目录获取样本ID
    if photo_dir.exists():
        for patient_dir in photo_dir.iterdir():
            if patient_dir.is_dir():
                sample_ids.add(patient_dir.name)

    # 从病理目录获取样本ID
    if pathology_dir.exists():
        for patient_dir in pathology_dir.iterdir():
            if patient_dir.is_dir():
                sample_ids.add(patient_dir.name)

    # 为每个样本创建索引条目
    for sample_id in sample_ids:
        sample = {
            "id": sample_id,
            "label": labels_dict.get(sample_id, "unknown"),
            "text_path": None,
            "photo_paths": [],
            "pathology_paths": [],
        }

        # 文本路径
        text_file = text_dir / f"{sample_id}.txt"
        if text_file.exists():
            sample["text_path"] = str(text_file.relative_to(data_path))

        # 照片路径
        sample_photo_dir = photo_dir / sample_id
        if sample_photo_dir.exists():
            for photo_file in sample_photo_dir.glob("*.png"):
                sample["photo_paths"].append(str(photo_file.relative_to(data_path)))
            for photo_file in sample_photo_dir.glob("*.jpg"):
                sample["photo_paths"].append(str(photo_file.relative_to(data_path)))
            for photo_file in sample_photo_dir.glob("*.jpeg"):
                sample["photo_paths"].append(str(photo_file.relative_to(data_path)))

        # 病理路径
        sample_pathology_dir = pathology_dir / sample_id
        if sample_pathology_dir.exists():
            for path_file in sample_pathology_dir.glob("*.tiff"):
                sample["pathology_paths"].append(
                    {
                        "path": str(path_file.relative_to(data_path)),
                        "roi": None,  # 可以后续添加ROI信息
                    }
                )
            for path_file in sample_pathology_dir.glob("*.tif"):
                sample["pathology_paths"].append(
                    {"path": str(path_file.relative_to(data_path)), "roi": None}
                )

        samples.append(sample)

    # 数据分割
    splits = create_data_splits(samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    # 保存索引文件
    for split_name, split_samples in splits.items():
        index_file = output_path / f"{split_name}_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(split_samples, f, ensure_ascii=False, indent=2)

        print(f"{split_name} 数据集: {len(split_samples)} 个样本 -> {index_file}")

    return splits


def create_data_splits(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    stratify: bool = True,
    random_state: int = 42,
) -> Dict[str, List[Dict]]:
    """
    创建数据分割

    Args:
        samples: 样本列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        stratify: 是否按标签分层
        random_state: 随机种子

    Returns:
        分割后的数据字典
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("分割比例之和必须等于1.0")

    # 检查是否可以使用分层采样
    if stratify and len(samples) > 0:
        from collections import Counter

        labels = [sample["label"] for sample in samples]
        label_counts = Counter(labels)

        # 检查是否有类别样本数少于2
        min_samples_per_class = min(label_counts.values())
        can_stratify = min_samples_per_class >= 2

        if not can_stratify:
            print(f"⚠️ 警告: 最少类别只有 {min_samples_per_class} 个样本，禁用分层采样")
            print(f"   类别分布: {dict(label_counts)}")
            stratify = False

    # 提取标签用于分层
    labels = [sample["label"] for sample in samples] if stratify else None

    # 处理样本数量很少的情况
    if len(samples) <= 3:
        print(f"⚠️ 样本数量很少 ({len(samples)}个)，简化分割策略")
        # 简单分割：第一个样本作为训练集，其余按比例分配
        if len(samples) == 1:
            return {"train": samples, "val": [], "test": []}
        elif len(samples) == 2:
            return {"train": [samples[0]], "val": [samples[1]], "test": []}
        else:  # len(samples) == 3
            return {"train": [samples[0]], "val": [samples[1]], "test": [samples[2]]}

    # 第一次分割：分离训练集和临时集（验证+测试）
    temp_ratio = val_ratio + test_ratio

    try:
        train_samples, temp_samples = train_test_split(
            samples,
            test_size=temp_ratio,
            random_state=random_state,
            stratify=labels if stratify else None,
        )
    except ValueError as e:
        if "least populated class" in str(e):
            print(f"⚠️ 分层采样失败，自动禁用分层采样: {e}")
            train_samples, temp_samples = train_test_split(
                samples, test_size=temp_ratio, random_state=random_state, stratify=None
            )
        else:
            raise e

    # 第二次分割：分离验证集和测试集
    if temp_ratio > 0 and len(temp_samples) > 0:
        val_size_in_temp = val_ratio / temp_ratio

        # 更新标签（如果使用分层）
        if stratify and len(temp_samples) > 1:
            temp_labels = [sample["label"] for sample in temp_samples]
            temp_label_counts = Counter(temp_labels)
            temp_can_stratify = min(temp_label_counts.values()) >= 2
        else:
            temp_can_stratify = False

        try:
            val_samples, test_samples = train_test_split(
                temp_samples,
                test_size=(1 - val_size_in_temp),
                random_state=random_state,
                stratify=temp_labels if (stratify and temp_can_stratify) else None,
            )
        except ValueError as e:
            if "least populated class" in str(e):
                print("⚠️ 第二次分割的分层采样失败，使用随机分割")
                val_samples, test_samples = train_test_split(
                    temp_samples,
                    test_size=(1 - val_size_in_temp),
                    random_state=random_state,
                    stratify=None,
                )
            else:
                raise e
    else:
        val_samples, test_samples = [], []

    splits = {"train": train_samples, "val": val_samples, "test": test_samples}

    # 打印分割统计
    print("数据分割统计:")
    for split_name, split_samples in splits.items():
        print(f"  {split_name}: {len(split_samples)} 样本")

        # 类别分布
        if split_samples:
            label_counts = Counter([sample["label"] for sample in split_samples])
            print(f"    类别分布: {dict(label_counts)}")

    return splits


def analyze_dataset(
    data_dir: str, index_files: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    分析数据集统计信息

    Args:
        data_dir: 数据目录
        index_files: 索引文件列表，如果为None则自动查找

    Returns:
        分析结果字典
    """
    data_path = Path(data_dir)

    # 查找索引文件
    if index_files is None:
        index_files = list(data_path.glob("*_index.json"))
    else:
        index_files = [Path(f) for f in index_files]

    analysis = {
        "total_samples": 0,
        "splits": {},
        "label_distribution": Counter(),
        "modality_availability": {"text": 0, "photo": 0, "pathology": 0},
        "modality_statistics": {
            "photos_per_sample": [],
            "pathology_per_sample": [],
            "text_lengths": [],
        },
    }

    # 分析每个分割
    for index_file in index_files:
        split_name = index_file.stem.replace("_index", "")

        with open(index_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        split_analysis = {
            "num_samples": len(samples),
            "label_counts": Counter(),
            "modality_counts": {"text": 0, "photo": 0, "pathology": 0},
        }

        for sample in samples:
            # 标签统计
            label = sample.get("label", "unknown")
            split_analysis["label_counts"][label] += 1
            analysis["label_distribution"][label] += 1

            # 模态可用性
            if sample.get("text_path"):
                split_analysis["modality_counts"]["text"] += 1
                analysis["modality_availability"]["text"] += 1

                # 文本长度统计
                try:
                    text_file = data_path / sample["text_path"]
                    if text_file.exists():
                        with open(text_file, "r", encoding="utf-8") as tf:
                            text_length = len(tf.read())
                            analysis["modality_statistics"]["text_lengths"].append(
                                text_length
                            )
                except:
                    pass

            if sample.get("photo_paths"):
                split_analysis["modality_counts"]["photo"] += 1
                analysis["modality_availability"]["photo"] += 1
                analysis["modality_statistics"]["photos_per_sample"].append(
                    len(sample["photo_paths"])
                )

            if sample.get("pathology_paths"):
                split_analysis["modality_counts"]["pathology"] += 1
                analysis["modality_availability"]["pathology"] += 1
                analysis["modality_statistics"]["pathology_per_sample"].append(
                    len(sample["pathology_paths"])
                )

        analysis["splits"][split_name] = split_analysis
        analysis["total_samples"] += len(samples)

    return analysis


def print_dataset_analysis(analysis: Dict[str, Any]):
    """打印数据集分析结果"""
    print(f"\n{'='*50}")
    print("数据集分析报告")
    print(f"{'='*50}")

    print(f"总样本数: {analysis['total_samples']}")

    # 分割统计
    print(f"\n分割统计:")
    for split_name, split_info in analysis["splits"].items():
        print(f"  {split_name}: {split_info['num_samples']} 样本")

        # 类别分布
        print(f"    类别分布:")
        for label, count in split_info["label_counts"].items():
            percentage = count / split_info["num_samples"] * 100
            print(f"      {label}: {count} ({percentage:.1f}%)")

        # 模态统计
        print(f"    模态覆盖:")
        for modality, count in split_info["modality_counts"].items():
            percentage = count / split_info["num_samples"] * 100
            print(f"      {modality}: {count} ({percentage:.1f}%)")

    # 全局标签分布
    print(f"\n全局标签分布:")
    for label, count in analysis["label_distribution"].items():
        percentage = count / analysis["total_samples"] * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # 模态统计
    print(f"\n模态统计:")
    stats = analysis["modality_statistics"]

    if stats["photos_per_sample"]:
        photos_stats = np.array(stats["photos_per_sample"])
        print(
            f"  照片数量/样本: 均值={photos_stats.mean():.1f}, "
            f"中位数={np.median(photos_stats):.1f}, "
            f"最大={photos_stats.max()}, 最小={photos_stats.min()}"
        )

    if stats["pathology_per_sample"]:
        path_stats = np.array(stats["pathology_per_sample"])
        print(
            f"  病理切片数量/样本: 均值={path_stats.mean():.1f}, "
            f"中位数={np.median(path_stats):.1f}, "
            f"最大={path_stats.max()}, 最小={path_stats.min()}"
        )

    if stats["text_lengths"]:
        text_stats = np.array(stats["text_lengths"])
        print(
            f"  文本长度: 均值={text_stats.mean():.0f}字符, "
            f"中位数={np.median(text_stats):.0f}字符, "
            f"最大={text_stats.max()}, 最小={text_stats.min()}"
        )

    print(f"{'='*50}\n")


def visualize_dataset_distribution(
    analysis: Dict[str, Any], save_path: Optional[str] = None
):
    """可视化数据集分布"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 分割分布饼图
    ax1 = axes[0, 0]
    split_sizes = [
        split_info["num_samples"] for split_info in analysis["splits"].values()
    ]
    split_labels = list(analysis["splits"].keys())

    ax1.pie(split_sizes, labels=split_labels, autopct="%1.1f%%", startangle=90)
    ax1.set_title("数据分割分布")

    # 2. 标签分布柱状图
    ax2 = axes[0, 1]
    labels = list(analysis["label_distribution"].keys())
    counts = list(analysis["label_distribution"].values())

    bars = ax2.bar(labels, counts)
    ax2.set_title("类别分布")
    ax2.set_xlabel("类别")
    ax2.set_ylabel("样本数")
    ax2.tick_params(axis="x", rotation=45)

    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
        )

    # 3. 模态可用性
    ax3 = axes[1, 0]
    modalities = list(analysis["modality_availability"].keys())
    availabilities = list(analysis["modality_availability"].values())

    bars = ax3.bar(modalities, availabilities)
    ax3.set_title("模态可用性")
    ax3.set_xlabel("模态")
    ax3.set_ylabel("样本数")

    for bar, count in zip(bars, availabilities):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
        )

    # 4. 照片数量分布直方图
    ax4 = axes[1, 1]
    if analysis["modality_statistics"]["photos_per_sample"]:
        ax4.hist(
            analysis["modality_statistics"]["photos_per_sample"],
            bins=20,
            alpha=0.7,
            edgecolor="black",
        )
        ax4.set_title("每个样本的照片数量分布")
        ax4.set_xlabel("照片数量")
        ax4.set_ylabel("样本数")
    else:
        ax4.text(
            0.5, 0.5, "无照片数据", ha="center", va="center", transform=ax4.transAxes
        )
        ax4.set_title("每个样本的照片数量分布")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"数据分布图已保存到: {save_path}")

    return fig


def prepare_data_splits(
    data_dir: str, output_dir: Optional[str] = None, **kwargs
) -> str:
    """
    准备数据分割的便捷函数

    Args:
        data_dir: 数据目录
        output_dir: 输出目录，如果为None则使用data_dir
        **kwargs: 其他参数传递给create_data_index_from_directory

    Returns:
        输出目录路径
    """
    if output_dir is None:
        output_dir = data_dir

    # 创建数据索引
    splits = create_data_index_from_directory(data_dir, output_dir, **kwargs)

    # 分析数据集
    analysis = analyze_dataset(output_dir)
    print_dataset_analysis(analysis)

    # 保存分析结果
    analysis_file = Path(output_dir) / "dataset_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        # 处理Counter对象和numpy类型
        serializable_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, Counter):
                serializable_analysis[key] = dict(value)
            elif isinstance(value, dict):
                serializable_analysis[key] = {
                    k: dict(v) if isinstance(v, Counter) else v
                    for k, v in value.items()
                }
            else:
                serializable_analysis[key] = value

        json.dump(serializable_analysis, f, ensure_ascii=False, indent=2, default=str)

    print(f"数据分析结果已保存到: {analysis_file}")

    return output_dir


if __name__ == "__main__":
    # 测试数据工具函数
    print("测试数据工具函数...")

    # 创建测试数据目录结构
    test_data_dir = Path("./test_data")
    test_data_dir.mkdir(exist_ok=True)

    # 创建子目录
    (test_data_dir / "texts").mkdir(exist_ok=True)
    (test_data_dir / "photos").mkdir(exist_ok=True)
    (test_data_dir / "pathology").mkdir(exist_ok=True)

    # 创建一些测试文件
    for i in range(10):
        # 文本文件
        with open(
            test_data_dir / "texts" / f"patient_{i:03d}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(f"患者{i}的病历文本内容...")

        # 照片目录
        patient_photo_dir = test_data_dir / "photos" / f"patient_{i:03d}"
        patient_photo_dir.mkdir(exist_ok=True)

        # 创建空的照片文件（仅用于测试）
        for j in range(np.random.randint(1, 4)):
            (patient_photo_dir / f"photo_{j}.png").touch()

    # 创建标签文件
    labels = {f"patient_{i:03d}": f"class_{i%3}" for i in range(10)}
    with open(test_data_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # 测试数据准备
    try:
        output_dir = prepare_data_splits(
            str(test_data_dir), label_file=str(test_data_dir / "labels.json")
        )
        print(f"测试数据准备完成: {output_dir}")

        # 清理测试数据
        shutil.rmtree(test_data_dir)
        print("测试数据已清理")

    except Exception as e:
        print(f"测试失败: {e}")
        # 清理
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)

    print("数据工具函数测试完成!")
